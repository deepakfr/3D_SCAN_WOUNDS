

import os, json, base64, io
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# Optional: nibabel for NIfTI
try:
    import nibabel as nib
    NIB_OK = True
except Exception:
    NIB_OK = False

import trimesh
from openai import OpenAI

st.set_page_config(page_title="OpenAI Predictor — NIfTI / OBJ / Image", layout="wide")
st.title("⚕️ OpenAI Outcome Predictor — NIfTI • OBJ/MTL • Image (Demo)")
st.caption("Uploads a scan/mesh/image → extracts features + views → asks an OpenAI vision model for a non-diagnostic risk distribution.")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    # Prefer environment var, but allow manual entry
    default_key = os.getenv("OPENAI_API_KEY", "")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", value=default_key if default_key else "")
    model_name = st.text_input("Model", value="gpt-4o-mini")
    seed = st.number_input("Seed", min_value=0, step=1, value=7)
    st.markdown("---")
    st.write("**Disclaimer**: Research/demo only — not a medical device. No medical advice.")

uploaded = st.file_uploader(
    "Upload a NIfTI (.nii/.nii.gz), an OBJ mesh (.obj), or a 2D image (.png/.jpg)",
    type=["nii","nii.gz","obj","png","jpg","jpeg"]
)

# ---------- Utilities ----------
def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def normalize01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    mn, mx = np.percentile(a, 1), np.percentile(a, 99)
    a = np.clip((a - mn) / (mx - mn + 1e-6), 0, 1)
    return a

def central_views_from_volume(v: np.ndarray):
    # v in [0,1], shape [D,H,W]
    D,H,W = v.shape
    sd, sh, sw = D//2, H//2, W//2
    axial   = (v[sd] * 255).astype(np.uint8)
    coronal = (v[:, sh, :] * 255).astype(np.uint8)
    sagittal= (v[:, :, sw] * 255).astype(np.uint8)
    return Image.fromarray(axial), Image.fromarray(coronal), Image.fromarray(sagittal)

def plot_volume(v: np.ndarray):
    # Coarse preview using Plotly’s Volume
    fig = go.Figure(data=go.Volume(
        value=v.flatten(),
        x=np.repeat(np.arange(v.shape[2]), v.shape[0]*v.shape[1]),
        y=np.tile(np.repeat(np.arange(v.shape[1]), v.shape[2]), v.shape[0]),
        z=np.tile(np.arange(v.shape[0]), v.shape[1]*v.shape[2]),
        opacity=0.05,
        isomin=float(np.quantile(v, 0.7)),
        isomax=float(v.max() if v.size else 1.0),
        surface_count=8,
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0))
    return fig

def features_from_nifti(file):
    if not NIB_OK:
        raise RuntimeError("nibabel is not installed. Run: pip install nibabel")
    img = nib.load(file)
    vol = img.get_fdata().astype(np.float32)
    zooms = img.header.get_zooms()[:3] if len(img.header.get_zooms())>=3 else (1.0,1.0,1.0)
    v = normalize01(vol)
    axial, coronal, sagittal = central_views_from_volume(v)
    stats = {
        "type": "nifti",
        "shape": list(vol.shape),
        "voxel_spacing_mm": [float(z) for z in zooms],
        "intensity": {
            "min": float(np.min(vol)),
            "max": float(np.max(vol)),
            "mean": float(np.mean(vol)),
            "std": float(np.std(vol)),
            "p10": float(np.percentile(vol, 10)),
            "p50": float(np.percentile(vol, 50)),
            "p90": float(np.percentile(vol, 90)),
        }
    }
    thresh = np.percentile(vol, 90)
    stats["naive_high_intensity_fraction"] = float(np.mean(vol >= thresh))
    return v, (axial, coronal, sagittal), stats

def projections_from_points(P, bins=256):
    # P: [N,3] points; return density projections as grayscale images
    if P.size == 0:
        Z = np.zeros((bins, bins), dtype=np.uint8)
        return Image.fromarray(Z), Image.fromarray(Z), Image.fromarray(Z)
    mn = P.min(axis=0); mx = P.max(axis=0); span = np.clip(mx - mn, 1e-6, None)
    Q = (P - mn) / span
    def hist2(a, b):
        H, _, _ = np.histogram2d(a, b, bins=bins, range=[[0,1],[0,1]])
        H = H.astype(np.float32)
        if H.max() > 0: H /= H.max()
        return (H * 255).astype(np.uint8)
    axial   = hist2(Q[:,0], Q[:,1])
    coronal = hist2(Q[:,0], Q[:,2])
    sagittal= hist2(Q[:,1], Q[:,2])
    return Image.fromarray(axial), Image.fromarray(coronal), Image.fromarray(sagittal)

def features_from_obj(file, uploaded_name="mesh.obj"):
    # Save file so trimesh can resolve .mtl if present next to it
    tmp_dir = "."
    mesh_path = os.path.join(tmp_dir, uploaded_name)
    with open(mesh_path, "wb") as f:
        f.write(file.getbuffer())
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    if mesh.is_empty:
        raise RuntimeError("Failed to load mesh (empty).")

    # Sample surface points for density projections
    try:
        P, _ = trimesh.sample.sample_surface(mesh, 50000)
    except Exception:
        P = mesh.vertices.copy()

    axial, coronal, sagittal = projections_from_points(P, bins=256)
    bbox = mesh.extents if hasattr(mesh, "extents") else np.array([0,0,0], dtype=np.float32)
    watertight = bool(mesh.is_watertight)
    vol = float(mesh.volume) if watertight and hasattr(mesh, "volume") else float("nan")
    area = float(mesh.area) if hasattr(mesh, "area") else float("nan")
    verts = int(mesh.vertices.shape[0])
    faces = int(mesh.faces.shape[0]) if hasattr(mesh, "faces") else 0
    compactness = float((36*np.pi*(vol**2))/(area**3)) if watertight and area>0 else float("nan")

    stats = {
        "type": "obj_mesh",
        "vertex_count": verts,
        "face_count": faces,
        "bbox_extents": [float(x) for x in bbox],
        "surface_area": area,
        "volume": vol,
        "watertight": watertight,
        "compactness": compactness
    }
    # Voxelize for coarse 3D preview
    try:
        pitch = max(bbox)/64 if max(bbox)>0 else 1.0
        vox = mesh.voxelized(pitch=pitch).matrix.astype(np.float32)
        v = (vox / (vox.max() if vox.max()>0 else 1.0))
    except Exception:
        v = np.zeros((32,32,32), dtype=np.float32)
    return v, (axial, coronal, sagittal), stats

def features_from_image(file):
    im = Image.open(file).convert("L")
    arr = np.array(im).astype(np.float32)
    v = normalize01(arr)
    # mimic small 3D stack for plotly preview
    stack = np.stack([v, v, v], axis=0).astype(np.float32)
    axial = Image.fromarray((v*255).astype(np.uint8))
    coronal = axial.resize((max(1, axial.width//2), max(1, axial.height//2)))
    sagittal = axial.resize((int(axial.width*0.75), int(axial.height*0.75)))
    stats = {
        "type": "image2d",
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "intensity": {
            "min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean()), "std": float(arr.std()),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90))
        }
    }
    thresh = np.percentile(arr, 90)
    stats["naive_high_intensity_fraction"] = float(np.mean(arr >= thresh))
    return stack, (axial, coronal, sagittal), stats

# ---------- Inference flow ----------
if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

suffix = os.path.splitext(uploaded.name.lower())[-1]
if suffix in [".nii", ".gz"]:
    mode = "nifti"
elif suffix in [".obj"]:
    mode = "obj"
elif suffix in [".png", ".jpg", ".jpeg"]:
    mode = "image"
else:
    st.error("Unsupported file type.")
    st.stop()

try:
    if mode == "nifti":
        v, (ax_img, co_img, sa_img), stats = features_from_nifti(uploaded)
    elif mode == "obj":
        v, (ax_img, co_img, sa_img), stats = features_from_obj(uploaded, uploaded_name=uploaded.name)
    else:
        v, (ax_img, co_img, sa_img), stats = features_from_image(uploaded)
except Exception as e:
    st.error(f"Failed to parse file: {e}")
    st.stop()

# Show views
st.subheader("Views")
c1,c2,c3 = st.columns(3)
with c1: st.image(ax_img, caption="View A", use_container_width=True)
with c2: st.image(co_img, caption="View B", use_container_width=True)
with c3: st.image(sa_img, caption="View C", use_container_width=True)

# Show coarse volume
if v.ndim == 3 and v.size:
    st.subheader("Volume preview")
    st.plotly_chart(plot_volume(v), use_container_width=True)

with st.expander("Computed summary features", expanded=False):
    st.json(stats)

# OpenAI call
if not openai_key:
    st.warning("Enter your OPENAI_API_KEY in the sidebar.")
    st.stop()

client = OpenAI(api_key=openai_key)

def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

ax_b64 = _b64(ax_img)
co_b64 = _b64(co_img)
sa_b64 = _b64(sa_img)

system_msg = {
    "role": "system",
    "content": (
        "You are a clinical research model that estimates a non-diagnostic outcome risk distribution "
        "from features and orthogonal views of a wound-related scan or mesh. "
        "Return valid JSON with keys: "
        "'outcome_probs' (dict with keys 'low','medium','high' summing to 1.0), "
        "'predicted' (one of 'low','medium','high'), "
        "'rationale' (<=120 words, plain text, non-diagnostic), "
        "'quality_flags' (list of strings). "
        "If inputs look unreliable, reflect uncertainty. Never provide medical advice."
    )
}

user_parts = [
    {"type": "text", "text": f"Modality: {stats.get('type','unknown')}\nFeatures (JSON):\n{json.dumps(stats)}\n\nReturn ONLY JSON (no markdown)."},
    {"type": "input_image", "image_url": {"url": ax_b64, "detail": "low"}},
    {"type": "input_image", "image_url": {"url": co_b64, "detail": "low"}},
    {"type": "input_image", "image_url": {"url": sa_b64, "detail": "low"}},
]

st.subheader("Predict with OpenAI")
with st.spinner("Calling OpenAI model…"):
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[system_msg, {"role": "user", "content": user_parts}],
            temperature=0.2,
            max_tokens=300,
            seed=int(seed),
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
    except Exception as e:
        st.error(f"OpenAI call failed: {e}")
        st.stop()

st.success(f"Predicted band: {result.get('predicted','unknown').upper()}")
colA, colB = st.columns(2)
with colA:
    st.markdown("**Outcome probabilities**")
    st.json(result.get("outcome_probs", {}))
with colB:
    st.markdown("**Quality flags**")
    st.json(result.get("quality_flags", []))

st.markdown("**Rationale (non-diagnostic)**")
st.write(result.get("rationale", ""))

st.caption("This is a research demonstration and not a medical device.")
