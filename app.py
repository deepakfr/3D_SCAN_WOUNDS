# app.py
# Streamlit app for 3D Scan Wounds – Outcome Predictor
# Uses OpenAI model with API key stored in st.secrets["OPENAI_API_KEY"]
# Supports: NIfTI (.nii/.nii.gz), OBJ (+MTL), and 2D images (.png/.jpg)
# Non-diagnostic demo for research purposes.

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


# ================================
# 🌟 App Configuration
# ================================
st.set_page_config(page_title="3D Scan Wounds – Outcome Predictor", layout="wide")
st.title("🩹 3D Scan Wounds – Outcome Predictor")
st.caption("Upload a 3D scan, mesh, or image — the app extracts basic features and sends them to an OpenAI vision model for a **non-diagnostic** outcome estimation.")

# Get OpenAI API key from secrets (Streamlit Cloud → Settings → Secrets)
if "OPENAI_API_KEY" not in st.secrets:
    st.error("⚠️ Missing `OPENAI_API_KEY` in st.secrets! Add it under Settings → Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("OpenAI Model", value="gpt-4o-mini")
    seed = st.number_input("Seed", min_value=0, step=1, value=7)
    st.markdown("---")
    st.write("**Disclaimer:** Research/demo only – not a medical device or diagnostic tool.")


# ================================
# 📤 Upload Section
# ================================
uploaded = st.file_uploader(
    "Upload a 3D NIfTI scan (.nii/.nii.gz), an OBJ mesh (.obj), or a 2D image (.png/.jpg)",
    type=["nii","nii.gz","obj","png","jpg","jpeg"]
)


# ================================
# 🔧 Utility Functions
# ================================
def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

def normalize01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    mn, mx = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - mn) / (mx - mn + 1e-6), 0, 1)

def central_views_from_volume(v: np.ndarray):
    D,H,W = v.shape
    sd, sh, sw = D//2, H//2, W//2
    axial   = (v[sd] * 255).astype(np.uint8)
    coronal = (v[:, sh, :] * 255).astype(np.uint8)
    sagittal= (v[:, :, sw] * 255).astype(np.uint8)
    return Image.fromarray(axial), Image.fromarray(coronal), Image.fromarray(sagittal)

def plot_volume(v: np.ndarray):
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


# ================================
# 📦 Feature Extraction per Type
# ================================
def features_from_nifti(file):
    if not NIB_OK:
        raise RuntimeError("nibabel not installed. Run: pip install nibabel")
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
        },
        "naive_high_intensity_fraction": float(np.mean(vol >= np.percentile(vol, 90)))
    }
    return v, (axial, coronal, sagittal), stats

def projections_from_points(P, bins=256):
    if P.size == 0:
        Z = np.zeros((bins, bins), dtype=np.uint8)
        return Image.fromarray(Z), Image.fromarray(Z), Image.fromarray(Z)
    mn, mx = P.min(axis=0), P.max(axis=0)
    Q = (P - mn) / np.clip(mx - mn, 1e-6, None)
    def hist2(a, b):
        H, _, _ = np.histogram2d(a, b, bins=bins, range=[[0,1],[0,1]])
        if H.max() > 0: H = H / H.max()
        return (H * 255).astype(np.uint8)
    return (
        Image.fromarray(hist2(Q[:,0], Q[:,1])),
        Image.fromarray(hist2(Q[:,0], Q[:,2])),
        Image.fromarray(hist2(Q[:,1], Q[:,2])),
    )

def features_from_obj(file, uploaded_name="mesh.obj"):
    tmp_path = os.path.join(".", uploaded_name)
    with open(tmp_path, "wb") as f:
        f.write(file.getbuffer())
    mesh = trimesh.load(tmp_path, force="mesh", process=False)
    if mesh.is_empty:
        raise RuntimeError("Empty or invalid mesh file.")

    try:
        P, _ = trimesh.sample.sample_surface(mesh, 50000)
    except Exception:
        P = mesh.vertices.copy()

    axial, coronal, sagittal = projections_from_points(P)
    bbox = mesh.extents
    watertight = bool(mesh.is_watertight)
    vol = float(mesh.volume) if watertight else float("nan")
    area = float(mesh.area)
    compactness = float((36*np.pi*(vol**2))/(area**3)) if watertight and area>0 else float("nan")
    stats = {
        "type": "obj_mesh",
        "vertex_count": int(mesh.vertices.shape[0]),
        "face_count": int(mesh.faces.shape[0]),
        "bbox_extents": [float(x) for x in bbox],
        "surface_area": area,
        "volume": vol,
        "watertight": watertight,
        "compactness": compactness
    }
    try:
        vox = mesh.voxelized(pitch=max(bbox)/64 if max(bbox)>0 else 1.0).matrix.astype(np.float32)
        v = (vox / (vox.max() if vox.max()>0 else 1.0))
    except Exception:
        v = np.zeros((32,32,32), dtype=np.float32)
    return v, (axial, coronal, sagittal), stats

def features_from_image(file):
    im = Image.open(file).convert("L")
    arr = np.array(im).astype(np.float32)
    v = normalize01(arr)
    stack = np.stack([v, v, v], axis=0)
    axial = Image.fromarray((v*255).astype(np.uint8))
    coronal = axial.resize((axial.width//2, axial.height//2))
    sagittal = axial.resize((int(axial.width*0.75), int(axial.height*0.75)))
    stats = {
        "type": "image2d",
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "intensity": {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90))
        },
        "naive_high_intensity_fraction": float(np.mean(arr >= np.percentile(arr, 90)))
    }
    return stack, (axial, coronal, sagittal), stats


# ================================
# 🔍 Detect Type and Extract
# ================================
if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

suffix = os.path.splitext(uploaded.name.lower())[-1]
mode = "nifti" if suffix in [".nii", ".gz"] else "obj" if suffix==".obj" else "image"

try:
    if mode == "nifti":
        v, (ax, co, sa), stats = features_from_nifti(uploaded)
    elif mode == "obj":
        v, (ax, co, sa), stats = features_from_obj(uploaded, uploaded_name=uploaded.name)
    else:
        v, (ax, co, sa), stats = features_from_image(uploaded)
except Exception as e:
    st.error(f"Failed to process file: {e}")
    st.stop()


# ================================
# 🖼️ Visualization
# ================================
st.subheader("Views")
cols = st.columns(3)
for c, (img, label) in zip(cols, [(ax,"View A"), (co,"View B"), (sa,"View C")]):
    with c: st.image(img, caption=label, use_container_width=True)

if v.ndim == 3 and v.size:
    st.subheader("Volume Preview")
    st.plotly_chart(plot_volume(v), use_container_width=True)

with st.expander("Computed Summary Features", expanded=False):
    st.json(stats)


# ================================
# 🤖 OpenAI Prediction
# ================================
st.subheader("Predict with OpenAI Model")

ax_b64, co_b64, sa_b64 = img_to_b64(ax), img_to_b64(co), img_to_b64(sa)

system_msg = {
    "role": "system",
    "content": (
        "You are a research model estimating *non-diagnostic* outcome risk from 3D wound scans or meshes. "
        "Return valid JSON with keys: 'outcome_probs' (dict low/medium/high summing to 1), "
        "'predicted' (low/medium/high), 'rationale' (<=120 words, plain text), "
        "and 'quality_flags' (list). Avoid any medical advice."
    )
}

user_parts = [
    {"type": "text", "text": f"Modality: {stats.get('type')}\nFeatures:\n{json.dumps(stats)}\nReturn ONLY JSON (no markdown)."},
    {"type": "input_image", "image_url": {"url": ax_b64, "detail": "low"}},
    {"type": "input_image", "image_url": {"url": co_b64, "detail": "low"}},
    {"type": "input_image", "image_url": {"url": sa_b64, "detail": "low"}},
]

with st.spinner("Querying OpenAI model..."):
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

st.success(f"Predicted Band: {result.get('predicted','unknown').upper()}")
col1, col2 = st.columns(2)
with col1: st.json(result.get("outcome_probs", {}))
with col2: st.json(result.get("quality_flags", []))
st.markdown("**Rationale (Non-diagnostic)**")
st.write(result.get("rationale", ""))
st.caption("⚠️ This is a research demonstration only, not a medical device.")
