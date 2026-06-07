import torch


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [int(b['lengths']) for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)


def video_t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    # word_embeddings, pos_one_hots, caption, sent_len, motion, camera, motion_2d, scores_2d, gt_motion, m_length, '_'.join(tokens)
    # 0	               1	         2	      3	        4	    5	    6          7	      8          9         10
    notnone_batches = [b for b in batch if b is not None]
    databatch = [torch.tensor(b[4].T).float().unsqueeze(1) for b in notnone_batches]
    lenbatch = [b[9] for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}, 'extra':{}}

    textbatch = [b[2] for b in notnone_batches]
    cond['y'].update({'text': textbatch})

    textbatch = [b[10] for b in notnone_batches]
    cond['y'].update({'tokens': textbatch})
    
    gt_motionbatch = [torch.tensor(b[8].T).float().unsqueeze(1) for b in notnone_batches]
    cond['extra']['gt_motion'] = collate_tensors(gt_motionbatch) # For cleaner setups, using gt teacher or supervision

    scores_2dbatch = [torch.tensor(b[7]).float() for b in notnone_batches]
    cond['extra']['scores_2d'] = collate_tensors(scores_2dbatch) # For loss

    motion_2dbatch = [torch.tensor(b[6]).float() for b in notnone_batches]
    cond['extra']['motion_2d'] = collate_tensors(motion_2dbatch) # For loss

    camerabatch = [torch.tensor(b[5]).float() for b in notnone_batches]
    cond['extra']['camera'] = collate_tensors(camerabatch) # For loss

    return motion, cond

