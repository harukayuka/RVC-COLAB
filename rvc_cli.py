import os
import sys
import argparse
import time
import shutil
import json
import logging
import traceback
from pathlib import Path
from subprocess import Popen, PIPE
from threading import Thread
from time import sleep

# Setup path agar sesuai dengan struktur RVC
now_dir = os.getcwd()
sys.path.append(now_dir)

# Load environment variables jika ada .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import modul internal RVC
try:
    from infer.modules.vc.modules import VC
    from infer.modules.uvr5.modules import uvr
    from infer.lib.train.process_ckpt import (
        change_info, extract_small_model, merge, show_info,
    )
    from i18n.i18n import I18nAuto
    from configs.config import Config
    from sklearn.cluster import MiniBatchKMeans
    import torch
    import numpy as np
    import faiss
    import fairseq
    import platform
except ImportError as e:
    print(f"Error: Gagal mengimpor modul RVC. Pastikan script dijalankan di direktori root RVC.")
    print(f"Detail: {e}")
    sys.exit(1)

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
i18n = I18nAuto()
config = Config()
vc = VC(config)

# Setup direktori TEMP
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp

# Deteksi GPU
ngpu = torch.cuda.device_count()
gpu_infos = []
if torch.cuda.is_available() and ngpu > 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_infos.append(f"{i}: {gpu_name}")
    gpu_info_str = "\n".join(gpu_infos)
    logger.info(f"GPU terdeteksi:\n{gpu_info_str}")
else:
    gpu_info_str = "Tidak ada GPU NVIDIA terdeteksi (CPU mode mungkin lambat)"
    logger.warning(gpu_info_str)

# --- Fungsi Bantu ---

def run_command(cmd, description=""):
    """Menjalankan command shell dan menunggu selesai."""
    logger.info(f"Menjalankan: {description}")
    logger.debug(f"Command: {cmd}")
    try:
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, cwd=now_dir)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            logger.error(f"Gagal: {description}")
            if stderr:
                logger.error(stderr.decode('utf-8', errors='ignore'))
            return False
        logger.info(f"Selesai: {description}")
        return True
    except Exception as e:
        logger.error(f"Exception saat menjalankan {description}: {e}")
        return False

def preprocess_data(trainset_dir, exp_dir, sr, n_p):
    """Langkah 1: Preprocessing dataset."""
    sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}
    sr_val = sr_dict.get(sr, 40000)
    
    os.makedirs(f"{now_dir}/logs/{exp_dir}", exist_ok=True)
    log_path = f"{now_dir}/logs/{exp_dir}/preprocess.log"
    
    cmd = (
        f'"{config.python_cmd}" infer/modules/train/preprocess.py '
        f'"{trainset_dir}" {sr_val} {n_p} "{now_dir}/logs/{exp_dir}" '
        f'{config.noparallel} {config.preprocess_per}'
    )
    return run_command(cmd, "Preprocessing Data")

def extract_f0_feature(exp_dir, n_p, f0method, if_f0, version19, gpus_rmvpe):
    """Langkah 2: Ekstraksi F0 dan Fitur."""
    os.makedirs(f"{now_dir}/logs/{exp_dir}", exist_ok=True)
    
    # Ekstraksi F0
    if if_f0:
        if f0method == "rmvpe_gpu":
            # Logika multi-GPU untuk RMVPE (sederhana: jalankan di GPU pertama)
            gpus_list = gpus_rmvpe.split("-") if gpus_rmvpe else ["0"]
            for idx, gpu_id in enumerate(gpus_list):
                cmd = (
                    f'"{config.python_cmd}" infer/modules/train/extract/extract_f0_rmvpe.py '
                    f'{len(gpus_list)} {idx} {gpu_id} "{now_dir}/logs/{exp_dir}" {config.is_half}'
                )
                # Jalankan paralel atau sekuensial tergantung kebutuhan. 
                # Untuk CLI sederhana, kita jalankan sekuensial atau hanya satu proses utama
                if idx == 0: # Fokus pada proses utama untuk CLI
                    if not run_command(cmd, "Ekstraksi F0 (RMVPE GPU)"):
                        return False
        else:
            cmd = (
                f'"{config.python_cmd}" infer/modules/train/extract/extract_f0_print.py '
                f'"{now_dir}/logs/{exp_dir}" {n_p} {f0method}'
            )
            if not run_command(cmd, "Ekstraksi F0"):
                return False
    else:
        logger.info("Model tanpa F0, melewati ekstraksi F0.")

    # Ekstraksi Fitur
    feature_type = "3_feature256" if version19 == "v1" else "3_feature768"
    cmd = (
        f'"{config.python_cmd}" infer/modules/train/extract/extract_feature_print.py '
        f'{config.device} 1 0 0 "{now_dir}/logs/{exp_dir}" {version19} {config.is_half}'
    )
    return run_command(cmd, "Ekstraksi Fitur")

def train_model(
    exp_dir, sr, if_f0, spk_id, save_epoch, total_epoch, batch_size,
    pretrained_G, pretrained_D, gpus, if_cache_gpu, if_save_every_weights, version19
):
    """Langkah 3: Training Model."""
    exp_log_dir = f"{now_dir}/logs/{exp_dir}"
    os.makedirs(exp_log_dir, exist_ok=True)

    # Buat filelist
    gt_wavs_dir = f"{exp_log_dir}/0_gt_wavs"
    feature_dir = f"{exp_log_dir}/{ '3_feature256' if version19 == 'v1' else '3_feature768' }"
    f0_dir = f"{exp_log_dir}/2a_f0"
    f0nsf_dir = f"{exp_log_dir}/2b-f0nsf"

    # Validasi file
    if not os.path.exists(gt_wavs_dir) or not os.path.exists(feature_dir):
        logger.error("File data tidak ditemukan. Pastikan preprocessing dan ekstraksi fitur sudah dilakukan.")
        return False

    names = set()
    if if_f0:
        if os.path.exists(f0_dir) and os.path.exists(f0nsf_dir):
            names = (
                set([n.split(".")[0] for n in os.listdir(gt_wavs_dir)]) &
                set([n.split(".")[0] for n in os.listdir(feature_dir)]) &
                set([n.split(".")[0] for n in os.listdir(f0_dir)]) &
                set([n.split(".")[0] for n in os.listdir(f0nsf_dir)])
            )
    else:
        names = (
            set([n.split(".")[0] for n in os.listdir(gt_wavs_dir)]) &
            set([n.split(".")[0] for n in os.listdir(feature_dir)])
        )

    if not names:
        logger.error("Tidak ada file audio yang valid ditemukan di direktori data.")
        return False

    # Tulis filelist
    opt = []
    for name in names:
        if if_f0:
            opt.append(
                f'{gt_wavs_dir.replace("\\", "\\\\")}/{name}.wav|'
                f'{feature_dir.replace("\\", "\\\\")}/{name}.npy|'
                f'{f0_dir.replace("\\", "\\\\")}/{name}.wav.npy|'
                f'{f0nsf_dir.replace("\\", "\\\\")}/{name}.wav.npy|{spk_id}'
            )
        else:
            opt.append(
                f'{gt_wavs_dir.replace("\\", "\\\\")}/{name}.wav|'
                f'{feature_dir.replace("\\", "\\\\")}/{name}.npy|{spk_id}'
            )
    
    # Tambah mute
    fea_dim = 256 if version19 == "v1" else 768
    mute_path = f"{now_dir}/logs/mute"
    if if_f0:
        for _ in range(2):
            opt.append(
                f'{mute_path}/0_gt_wavs/mute{sr}.wav|'
                f'{mute_path}/3_feature{fea_dim}/mute.npy|'
                f'{mute_path}/2a_f0/mute.wav.npy|'
                f'{mute_path}/2b-f0nsf/mute.wav.npy|{spk_id}'
            )
    else:
        for _ in range(2):
            opt.append(
                f'{mute_path}/0_gt_wavs/mute{sr}.wav|'
                f'{mute_path}/3_feature{fea_dim}/mute.npy|{spk_id}'
            )

    with open(f"{exp_log_dir}/filelist.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(opt))

    # Konfigurasi training
    config_path = f"v1/{sr}.json" if (version19 == "v1" or sr == "40k") else f"v2/{sr}.json"
    config_save_path = os.path.join(exp_log_dir, "config.json")
    
    if not os.path.exists(config_save_path):
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config.json_config.get(config_path, {}), f, ensure_ascii=False, indent=4)

    # Build command training
    f0_flag = 1 if if_f0 else 0
    save_latest_flag = 1 # Default true untuk CLI sederhana
    cache_gpu_flag = 1 if if_cache_gpu else 0
    save_weights_flag = 1 if if_save_every_weights else 0

    cmd = (
        f'"{config.python_cmd}" infer/modules/train/train.py '
        f'-e "{exp_dir}" -sr {sr} -f0 {f0_flag} -bs {batch_size} -g "{gpus}" '
        f'-te {total_epoch} -se {save_epoch} '
        f'{"-pg " + pretrained_G if pretrained_G else ""} '
        f'{"-pd " + pretrained_D if pretrained_D else ""} '
        f'-l {save_latest_flag} -c {cache_gpu_flag} -sw {save_weights_flag} -v {version19}'
    )

    return run_command(cmd, "Training Model")

def train_index(exp_dir, version19):
    """Langkah 4: Training Index FAISS."""
    exp_log_dir = f"{now_dir}/logs/{exp_dir}"
    feature_dir = f"{exp_log_dir}/{ '3_feature256' if version19 == 'v1' else '3_feature768' }"
    
    if not os.path.exists(feature_dir):
        logger.error("Direktori fitur tidak ditemukan.")
        return False

    npys = []
    for name in sorted(os.listdir(feature_dir)):
        phone = np.load(f"{feature_dir}/{name}")
        npys.append(phone)
    
    if not npys:
        logger.error("Tidak ada data fitur untuk di-index.")
        return False

    big_npy = np.concatenate(npys, 0)
    logger.info(f"Total fitur: {big_npy.shape[0]}")

    # KMeans jika data besar
    if big_npy.shape[0] > 2e5:
        logger.info("Menggunakan KMeans untuk reduksi data...")
        big_npy = (
            MiniBatchKMeans(n_clusters=10000, verbose=False, batch_size=256 * config.n_cpu, compute_labels=False, init="random")
            .fit(big_npy)
            .cluster_centers_
        )

    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    index = faiss.index_factory(256 if version19 == "v1" else 768, f"IVF{n_ivf},Flat")
    index.train(big_npy)
    
    index_path_trained = f"{exp_log_dir}/trained_IVF{n_ivf}_Flat_nprobe_1_{exp_dir}_{version19}.index"
    faiss.write_index(index, index_path_trained)

    logger.info("Menambah data ke index...")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    
    index_path_final = f"{exp_log_dir}/added_IVF{n_ivf}_Flat_nprobe_1_{exp_dir}_{version19}.index"
    faiss.write_index(index, index_path_final)
    
    logger.info(f"Index berhasil disimpan: {index_path_final}")
    return True

def infer_single(voice_path, input_audio, output_audio, model_path, index_path, f0_method, f0_up_key, protect):
    """Inferensi satu file."""
    if not os.path.exists(model_path):
        logger.error(f"Model tidak ditemukan: {model_path}")
        return False
    
    # Load model ke VC
    try:
        vc.get_vc(voice_path, protect, protect, "", "") # Load model
    except Exception as e:
        logger.error(f"Gagal memuat model: {e}")
        return False

    # Jalankan inferensi
    try:
        # vc.vc_single butuh banyak argumen. Kita panggil langsung.
        # spk_item, input_audio0, vc_transform0, f0_file, f0method0, file_index1, file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect
        _, audio_output = vc.vc_single(
            spk_item=0,
            input_audio0=input_audio,
            vc_transform0=f0_up_key,
            f0_file=None,
            f0method0=f0_method,
            file_index1=index_path if os.path.exists(index_path) else "",
            file_index2="",
            index_rate1=0.75,
            filter_radius0=3,
            resample_sr0=0,
            rms_mix_rate0=0.25,
            protect0=protect
        )
        
        if audio_output:
            # Simpan audio output
            # audio_output biasanya dalam format numpy array atau path, tergantung implementasi VC
            # Di infer-web.py, output adalah Audio component, jadi kita perlu menangani formatnya
            # Asumsi vc.vc_single mengembalikan (info, audio_path) atau (info, audio_data)
            # Kita asumsikan mengembalikan path jika sudah diproses, atau kita simpan manual jika array
            
            # Catatan: Implementasi asli vc.vc_single mengembalikan (text, audio_path)
            # Jika audio_path kosong, kita simpan manual
            if isinstance(audio_output, str) and os.path.exists(audio_output):
                shutil.copy(audio_output, output_audio)
            else:
                # Jika berupa array (bisa terjadi di versi tertentu), simpan dengan scipy
                try:
                    from scipy.io import wavfile
                    sr = 40000 # Default, bisa disesuaikan dari model
                    wavfile.write(output_audio, sr, audio_output)
                except:
                    logger.error("Gagal menyimpan audio output.")
                    return False
            
            logger.info(f"Inferensi selesai: {output_audio}")
            return True
        else:
            logger.error("Inferensi gagal menghasilkan output.")
            return False
    except Exception as e:
        logger.error(f"Error saat inferensi: {e}")
        traceback.print_exc()
        return False

# --- Main CLI ---

def main():
    parser = argparse.ArgumentParser(description="RVC CLI Tool untuk Inferensi dan Pelatihan")
    subparsers = parser.add_subparsers(dest="command", help="Pilih perintah: train, infer, uvr")

    # --- Perintah Train ---
    train_parser = subparsers.add_parser("train", help="Melatih model RVC")
    train_parser.add_argument("--name", type=str, required=True, help="Nama eksperimen (folder logs)")
    train_parser.add_argument("--data", type=str, required=True, help="Folder dataset audio")
    train_parser.add_argument("--sr", type=str, default="40k", choices=["32k", "40k", "48k"], help="Sample rate")
    train_parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"], help="Versi model")
    train_parser.add_argument("--f0", action="store_true", default=True, help="Gunakan F0 (pitch)")
    train_parser.add_argument("--epochs", type=int, default=20, help="Total epoch")
    train_parser.add_argument("--batch", type=int, default=4, help="Batch size")
    train_parser.add_argument("--gpu", type=str, default="0", help="GPU ID (misal: 0 atau 0-1)")
    train_parser.add_argument("--pretrained-g", type=str, default="assets/pretrained_v2/f0G40k.pth", help="Path pretrained G")
    train_parser.add_argument("--pretrained-d", type=str, default="assets/pretrained_v2/f0D40k.pth", help="Path pretrained D")

    # --- Perintah Infer ---
    infer_parser = subparsers.add_parser("infer", help="Inferensi suara")
    infer_parser.add_argument("--model", type=str, required=True, help="Path file model (.pth)")
    infer_parser.add_argument("--index", type=str, default="", help="Path file index (.index)")
    infer_parser.add_argument("--input", type=str, required=True, help="Path file audio input")
    infer_parser.add_argument("--output", type=str, required=True, help="Path file audio output")
    infer_parser.add_argument("--pitch", type=int, default=0, help="Ubah pitch (semitone)")
    infer_parser.add_argument("--method", type=str, default="rmvpe", choices=["pm", "harvest", "crepe", "rmvpe"], help="Metode F0")
    infer_parser.add_argument("--protect", type=float, default=0.33, help="Proteksi suara (0.0 - 0.5)")

    # --- Perintah UVR ---
    uvr_parser = subparsers.add_parser("uvr", help="Pisahkan vokal dan instrumen")
    uvr_parser.add_argument("--model", type=str, default="HP2_all_vocals", help="Nama model UVR5")
    uvr_parser.add_argument("--input", type=str, required=True, help="Path file audio atau folder")
    uvr_parser.add_argument("--output", type=str, default="opt", help="Folder output")

    args = parser.parse_args()

    if args.command == "train":
        logger.info(f"Memulai pelatihan model: {args.name}")
        
        # 1. Preprocess
        if not preprocess_data(args.data, args.name, args.sr, 1):
            return
        
        # 2. Extract F0 & Feature
        if not extract_f0_feature(args.name, 1, args.method if hasattr(args, 'method') else "rmvpe", args.f0, args.version, args.gpu):
            return

        # 3. Train Model
        if not train_model(
            args.name, args.sr, args.f0, 0, 5, args.epochs, args.batch,
            args.pretrained_g, args.pretrained_d, args.gpu, False, False, args.version
        ):
            return

        # 4. Train Index
        train_index(args.name, args.version)
        logger.info("Pelatihan selesai!")

    elif args.command == "infer":
        logger.info(f"Memulai inferensi: {args.input} -> {args.output}")
        infer_single(
            voice_path=args.model,
            input_audio=args.input,
            output_audio=args.output,
            model_path=args.model,
            index_path=args.index,
            f0_method=args.method,
            f0_up_key=args.pitch,
            protect=args.protect
        )

    elif args.command == "uvr":
        logger.info(f"Memulai UVR5: {args.input}")
        # Implementasi UVR5 sederhana (memanggil fungsi uvr langsung)
        # Fungsi uvr di RVC asli cukup kompleks dengan UI, kita panggil dengan argumen dasar
        try:
            # uvr(model_name, input_dir, output_dir, agg, format, ...)
            # Ini adalah wrapper sederhana, mungkin perlu penyesuaian lebih lanjut tergantung versi RVC
            logger.warning("Perintah UVR5 CLI mungkin memerlukan penyesuaian lebih lanjut tergantung versi RVC.")
            # Placeholder untuk logika UVR
            print("Fitur UVR5 CLI belum diimplementasikan penuh dalam script ini.")
        except Exception as e:
            logger.error(f"Error UVR: {e}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()