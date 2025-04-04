import torch
import copy
import torch.nn.functional as F
import numpy as np


class ScoreParams:

    def __init__(self, gap, match, mismatch):
        self.gap = gap
        self.match = match
        self.mismatch = mismatch

    def mis_match_char(self, x, y):
        if x != y:
            return self.mismatch
        else:
            return self.match


def get_matrix(size_x, size_y, gap):
    matrix = []
    for i in range(len(size_x) + 1):
        sub_matrix = []
        for j in range(len(size_y) + 1):
            sub_matrix.append(0)
        matrix.append(sub_matrix)
    for j in range(1, len(size_y) + 1):
        matrix[0][j] = j*gap
    for i in range(1, len(size_x) + 1):
        matrix[i][0] = i*gap
    return matrix


def get_matrix(size_x, size_y, gap):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap
    return matrix


def get_traceback_matrix(size_x, size_y):
    matrix = np.zeros((size_x + 1, size_y +1), dtype=np.int32)
    matrix[0, 1:] = 1
    matrix[1:, 0] = 2
    matrix[0, 0] = 4
    return matrix


def global_align(x, y, score):
    matrix = get_matrix(len(x), len(y), score.gap)
    trace_back = get_traceback_matrix(len(x), len(y))
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            left = matrix[i, j - 1] + score.gap
            up = matrix[i - 1, j] + score.gap
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])
            matrix[i, j] = max(left, up, diag)
            if matrix[i, j] == left:
                trace_back[i, j] = 1
            elif matrix[i, j] == up:
                trace_back[i, j] = 2
            else:
                trace_back[i, j] = 3
    return matrix, trace_back


def get_aligned_sequences(x, y, trace_back):
    x_seq = []
    y_seq = []
    i = len(x)
    j = len(y)
    mapper_y_to_x = []
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:
            x_seq.append(x[i-1])
            y_seq.append(y[j-1])
            i = i-1
            j = j-1
            mapper_y_to_x.append((j, i))
        elif trace_back[i][j] == 1:
            x_seq.append('-')
            y_seq.append(y[j-1])
            j = j-1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2:
            x_seq.append(x[i-1])
            y_seq.append('-')
            i = i-1
        elif trace_back[i][j] == 4:
            break
    mapper_y_to_x.reverse()
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)


def get_mapper(x: str, y: str, specifier, tokenizer, encoder, device, max_len=77):
    locol_prompt, mutual_prompt = specifier
    x_seq = tokenizer.encode(x)
    y_seq = tokenizer.encode(y)
    e_seq = tokenizer.encode(locol_prompt)
    m_seq = tokenizer.encode(mutual_prompt)
    score = ScoreParams(0, 1, -1)
    matrix, trace_back = global_align(x_seq, y_seq, score)
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]
    alphas = torch.ones(max_len)
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()
    mapper = torch.zeros(max_len, dtype=torch.int64)
    mapper[:mapper_base.shape[0]] = mapper_base[:, 1]
    mapper[mapper_base.shape[0]:] = len(y_seq) + torch.arange(max_len - len(y_seq))
    m = copy.deepcopy(alphas)
    alpha_e = torch.zeros_like(alphas)
    alpha_m = torch.zeros_like(alphas)

    # print("mapper of")
    # print("<begin> "+x+" <end>")
    # print("<begin> "+y+" <end>")
    # print(mapper[:len(y_seq)])
    # print(alphas[:len(y_seq)])

    x = tokenizer(
            x,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
    y = tokenizer(
            y,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

    x_latent = encoder(x)[0].squeeze(0)
    y_latent = encoder(y)[0].squeeze(0)
    i = 0
    while i<len(y_seq):
        start = None
        if alphas[i] == 0:
            start = i
            while alphas[i] == 0:
                i += 1
            max_sim = float('-inf')
            max_s = None
            max_t = None
            for i_target in range(start, i):
                for i_source in range(mapper[start-1]+1, mapper[i]):
                    sim = F.cosine_similarity(x_latent[i_target], y_latent[i_source], dim=0)
                    if sim > max_sim:
                        max_sim = sim
                        max_s = i_source
                        max_t = i_target
            if max_s is not None:
                mapper[max_t] = max_s
                alphas[max_t] = 1
                for t in e_seq:
                  if x_seq[max_s] == t:
                    alpha_e[max_t] = 1
        i += 1

    # replace_alpha, replace_mapper = get_replace_inds(x_seq, y_seq, m_seq, m_seq)
    # if replace_mapper != []:
    #     mapper[replace_alpha]=torch.tensor(replace_mapper,device=mapper.device)
    #     alpha_m[replace_alpha]=1

    i = 1
    j = 1
    while (i < len(y_seq)-1) and (j < len(e_seq)-1):
        found = True
        while e_seq[j] != y_seq[i]:
            i = i + 1
            if i >= len(y_seq)-1:
                print("blend word not found!")
                found = False
                break
                raise ValueError("local prompt not found in target prompt")
        if found:
            alpha_e[i] = 1
        j = j + 1

    i = 1
    j = 1
    while (i < len(y_seq)-1) and (j < len(m_seq)-1):
      while m_seq[j] != y_seq[i]:
        i = i + 1
      if m_seq[j] == x_seq[mapper[i]]:
        alpha_m[i] = 1
        j = j + 1
      else:
        raise ValueError("mutual prompt not found in target prompt")

    # print("fixed mapper:")
    # print(mapper[:len(y_seq)])
    # print(alphas[:len(y_seq)])
    # print(m[:len(y_seq)])
    # print(alpha_e[:len(y_seq)])
    # print(alpha_m[:len(y_seq)])
    return mapper, alphas, m, alpha_e, alpha_m


def get_refinement_mapper(prompts, specifiers, tokenizer, encoder, device, max_len=77):
    x_seq = prompts[0]
    mappers, alphas, ms, alpha_objs, alpha_descs = [], [], [], [], []
    for i in range(1, len(prompts)):
        mapper, alpha, m, alpha_obj, alpha_desc = get_mapper(x_seq, prompts[i], specifiers[i-1], tokenizer, encoder, device, max_len)
        mappers.append(mapper)
        alphas.append(alpha)
        ms.append(m)
        alpha_objs.append(alpha_obj)
        alpha_descs.append(alpha_desc)
    return torch.stack(mappers), torch.stack(alphas), torch.stack(ms),  torch.stack(alpha_objs), torch.stack(alpha_descs)


def get_replace_inds(x_seq,y_seq,source_replace_seq,target_replace_seq):
    replace_mapper=[]
    replace_alpha=[]
    source_found=False
    source_match,target_match=[],[]
    for j in range(len(x_seq)):
        found=True
        for i in range(1,len(source_replace_seq)-1):
            if x_seq[j+i-1]!=source_replace_seq[i]:
                found=False
                break
        if found:
            source_found=True
            for i in range(1,len(source_replace_seq)-1):
                source_match.append(j+i-1)
    for j in range(len(y_seq)):
        found=True
        for i in range(1,len(target_replace_seq)-1):
            if y_seq[j+i-1]!=target_replace_seq[i]:
                found=False
                break
        if found:
            for i in range(1,len(source_replace_seq)-1):
                target_match.append(j+i-1)
    if not source_found:
        raise ValueError("replacing object not found in prompt")
    if (len(source_match)!=len(target_match)):
        raise ValueError(f"the replacement word number doesn't match for word {i}!")
    replace_alpha+=source_match
    replace_mapper+=target_match
    return replace_alpha,replace_mapper



def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
    words_x = x.split(' ')
    words_y = y.split(' ')
    if len(words_x) != len(words_y):
        raise ValueError(f"attention replacement edit can only be applied on prompts with the same length"
                         f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words.")
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]
    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    return torch.from_numpy(mapper).float()



def get_replacement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
    return torch.stack(mappers)

