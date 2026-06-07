import os
import glob
import wandb
import json

class TrainPlatform:
    def __init__(self, save_dir, *args, **kwargs):
        self.path, file = os.path.split(save_dir)
        self.name = kwargs.get('name', file)

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_media(self, title, series, iteration, local_path):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass

# Deprecated
class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from clearml import Task
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='motion_diffusion',
                              task_name=name,
                              output_uri=path)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_media(self, title, series, iteration, local_path):
        self.logger.report_media(title=title, series=series, iteration=iteration, local_path=local_path)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir, *args, **kwargs):
        pass

class WandBPlatform(TrainPlatform):
    WANDB_PROJET_NAME = 'video_mdm' # Alter this to the required project name.
    WANDB_PROJET_ENTITY = 'video_mdm' # Alter this to the required entity.
    def __init__(self, save_dir, config=None, *args, **kwargs):
        super().__init__(save_dir, *args, **kwargs)
        wandb.login(host=os.getenv("WANDB_BASE_URL"), key=os.getenv("WANDB_API_KEY"))
        self.wandb = wandb.init(
            project=self.WANDB_PROJET_NAME,
            name=self.name,
            id=self.name,  # in order to send continued runs to the same record
            resume='allow',  # in order to send continued runs to the same record
            #entity=self.WANDB_PROJET_ENTITY,  # will use your default entity if not set
            save_code=True,
            config=config
        )  # config can also be sent via report_args()

    def report_scalar(self, name, value, iteration, group_name=None):
        wandb.log({name: value}, step=iteration)

    def report_media(self, title, series, iteration, local_path):
        files = glob.glob(f'{local_path}/*.mp4')
        wandb.log({series: [wandb.Video(file, format='mp4', fps=20) for file in files]}, step=iteration)

    def report_args(self, args, name):
        wandb.config.update(args, allow_val_change=True)  # use allow_val_change ONLY if you want to change existing args (e.g., overwrite)

    def watch_model(self, *args, **kwargs):
        wandb.watch(args, kwargs)

    def close(self):
        wandb.finish()


class WandBSweepPlatform(TrainPlatform):
    # This class should be used on a different slurm node and report to the demon process via a log file.
    WANDB_REPORT_FILE_NAME = "wandb_report.log"
    WANDB_REPORT_FILE_END_LINE = "EXECUTION_FINISHED"
    def __init__(self, save_dir, config=None, *args, **kwargs):
        super().__init__(save_dir, *args, **kwargs)
        self.log_file_path = os.path.join(save_dir, self.WANDB_REPORT_FILE_NAME)
        self.log_file = open(self.log_file_path, "w")

    def report_scalar(self, name, value, iteration, group_name=None):
        self.log_file.write(json.dumps({name: float(value), "iteration": iteration}) + "\n")
        self.log_file.flush()

    def report_media(self, title, series, iteration, local_path):
        pass

    def report_args(self, args, name):
        args_name = f"args_{name}"
        self.log_file.write(json.dumps({args_name: vars(args)}) + "\n")
        self.log_file.flush()

    def watch_model(self, *args, **kwargs):
        pass

    def close(self):
        self.log_file.write(self.WANDB_REPORT_FILE_END_LINE + "\n")
        self.log_file.flush()
        self.log_file.close()

