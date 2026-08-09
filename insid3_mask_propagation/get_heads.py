import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict



# =========================================================
# Model loading
# =========================================================

_HUB_NAMES = {
    "small": "dinov3_vits16",
    "base": "dinov3_vitb16",
    "large": "dinov3_vitl16",
}

_WEIGHTS = {
    "small": "../checkpoints/dinov3_vits16.pth",
    "base": "../checkpoints/dinov3_vitb16.pth",
    "large": "../checkpoints/dinov3_vitl16.pth",
}


def get_device():

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"



def load_dinov3(
    model_size="base"
):

    device = get_device()

    model = torch.hub.load(
        "facebookresearch/dinov3",
        _HUB_NAMES[model_size],
        weights=_WEIGHTS[model_size],
    )

    model = model.to(device)
    model.eval()

    return model, device



# =========================================================
# Image preprocessing
# =========================================================

def preprocess_image(
    image,
    device,
    size=(1024,512)
):

    image = image.resize(size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225)
        )
    ])

    tensor = transform(image)

    tensor = tensor.unsqueeze(0).to(device)

    return tensor, image



# =========================================================
# QKV extraction
# =========================================================

def extract_qkv(
    model,
    image_tensor
):

    qkv_outputs = {}
    handles = []


    def hook_fn(layer_id):

        def hook(module, inputs, output):

            x = inputs[0]

            qkv = module.qkv(x)

            q,k,v = qkv.chunk(
                3,
                dim=-1
            )

            qkv_outputs[layer_id] = {
                "q": q.detach().cpu(),
                "k": k.detach().cpu(),
                "v": v.detach().cpu(),
            }


        return hook



    for idx, block in enumerate(model.blocks):

        h = block.attn.register_forward_hook(
            hook_fn(idx)
        )

        handles.append(h)



    with torch.no_grad():

        model.forward_features(
            image_tensor
        )


    for h in handles:
        h.remove()


    return qkv_outputs



# =========================================================
# Reshape QKV
# =========================================================

def reshape_qkv(
    qkv_outputs,
    model,
    H,
    W
):

    patch_size = model.patch_size

    hf = H // patch_size
    wf = W // patch_size

    n_patches = hf * wf


    num_heads = model.num_heads
    head_dim = model.embed_dim // num_heads


    special_tokens = (
        qkv_outputs[0]["q"].shape[1]
        -
        n_patches
    )


    processed = {}


    for layer, data in qkv_outputs.items():


        q = data["q"][:, special_tokens:]
        k = data["k"][:, special_tokens:]
        v = data["v"][:, special_tokens:]


        B = q.shape[0]


        q = q.reshape(
            B,n_patches,num_heads,head_dim
        ).permute(0,2,1,3)


        k = k.reshape(
            B,n_patches,num_heads,head_dim
        ).permute(0,2,1,3)


        v = v.reshape(
            B,n_patches,num_heads,head_dim
        ).permute(0,2,1,3)



        processed[layer] = {
            "q":q,
            "k":k,
            "v":v
        }


    return processed



# =========================================================
# Object DINO similarity
# =========================================================

def object_dino_similarity(
    x,
    temp=None
):

    d = x.shape[-1]

    if temp is None:
        temp = d ** -0.5


    x = F.normalize(
        x,
        dim=-1
    )


    logits = torch.einsum(
        "bhid,bhjd->bhij",
        x,
        x
    )


    attn = (
        logits * temp
    ).softmax(-1)


    return attn



def compute_object_maps(
    qkv
):

    output = {}


    for layer, data in qkv.items():

        output[layer] = {

            "qq":
            object_dino_similarity(
                data["q"]
            ),

            "kk":
            object_dino_similarity(
                data["k"]
            ),

            "vv":
            object_dino_similarity(
                data["v"]
            )

        }


    return output



# =========================================================
# Cluster attention heads
# =========================================================

def cluster_attention_heads(
    similarities,
    k_clusters=5
):

    maps=[]
    labels=[]


    for layer,data in similarities.items():

        attn = (
            data["qq"]
            +
            data["kk"]
            +
            data["vv"]
        ) / 3


        for head in range(
            attn.shape[1]
        ):

            token_map = (
                attn[0,head]
                .mean(0)
            )


            maps.append(
                token_map.numpy()
            )

            labels.append(
                (layer,head)
            )


    X = np.array(maps)


    X = StandardScaler().fit_transform(
        X
    )


    ids = KMeans(
        n_clusters=k_clusters,
        random_state=42,
        n_init=10
    ).fit_predict(X)



    clusters = {
        i:[]
        for i in range(k_clusters)
    }


    for label,c in zip(labels,ids):

        clusters[c].append(label)



    formatted={}


    for c,items in clusters.items():

        d=defaultdict(list)

        for layer,head in items:
            d[layer].append(head)


        formatted[c]=[
            (l,sorted(h))
            for l,h in sorted(d.items())
        ]


    return formatted



# =========================================================
# Select object cluster
# =========================================================

def select_object_cluster(
    clusters
):

    final_layer = max(
        l
        for c in clusters.values()
        for l,_ in c
    )


    best=None
    count=0


    for cid,items in clusters.items():

        for layer,heads in items:

            if layer==final_layer:

                if len(heads)>count:

                    best=cid
                    count=len(heads)


    return best, clusters[best]



# =========================================================
# MAIN FUNCTION
# =========================================================

def dinov3_attention_clustering(
    image,
    model_size="base",
    k_clusters=5
):

    model,device = load_dinov3(
        model_size
    )


    tensor,img = preprocess_image(
        image,
        device
    )


    H,W = img.size[1], img.size[0]


    qkv = extract_qkv(
        model,
        tensor
    )


    qkv = reshape_qkv(
        qkv,
        model,
        H,
        W
    )


    similarities = compute_object_maps(
        qkv
    )


    clusters = cluster_attention_heads(
        similarities,
        k_clusters
    )


    cluster_id,selections = select_object_cluster(
        clusters
    )


    return {
        "selected_cluster":cluster_id,
        "selections":selections,
        "clusters":clusters,
        "similarities":similarities
    }