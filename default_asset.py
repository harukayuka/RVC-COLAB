import os
import subprocess
import sys

def run_aria2c(url, dest_dir, filename):
    """
    Helper function untuk menjalankan perintah aria2c.
    """
    # Pastikan direktori tujuan ada
    os.makedirs(dest_dir, exist_ok=True)
    
    cmd = [
        "aria2c",
        "--console-log-level=error",
        "-c",          # Resume download jika terputus
        "-x", "16",    # 16 koneksi
        "-s", "16",    # 16 segment
        "-k", "1M",    # Minimum segment size 1M
        "-d", dest_dir,
        "-o", filename,
        url
    ]
    
    print(f"Mengunduh: {filename} ke {dest_dir}...")
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Berhasil mengunduh: {filename}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Gagal mengunduh {filename}: {e}")
        # Lanjutkan ke file berikutnya, jangan stop total

def main():
    # Tentukan direktori root proyek (asumsi script dijalankan dari root)
    # Jika dijalankan dari Colab, path biasanya /content/Retrieval-based-Voice-Conversion-WebUI
    # Kita coba deteksi atau gunakan path relatif jika sudah di dalam folder
    base_dir = os.getcwd()
    
    # Jika script ini ada di dalam folder RVC, base_dir sudah benar.
    # Jika tidak, kita cek apakah ada folder 'pretrained' di sini.
    if not os.path.exists(os.path.join(base_dir, "pretrained")):
        # Coba cek parent directory jika script dipindah
        parent = os.path.dirname(base_dir)
        if os.path.exists(os.path.join(parent, "pretrained")):
            base_dir = parent
            print(f"Mendeteksi direktori root di: {base_dir}")
        else:
            print(f"Peringatan: Direktori 'pretrained' tidak ditemukan di {base_dir}. Pastikan script dijalankan di root RVC.")

    # --- URL Base ---
    huggingface_base = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"

    # --- 1. Download Pretrained Models (v1) ---
    print("\n=== Mengunduh Pretrained Models V1 ===")
    pretrained_dir = os.path.join(base_dir, "pretrained")
    
    v1_models = [
        ("D32k.pth", f"{huggingface_base}/pretrained/D32k.pth"),
        ("D40k.pth", f"{huggingface_base}/pretrained/D40k.pth"),
        ("D48k.pth", f"{huggingface_base}/pretrained/D48k.pth"),
        ("G32k.pth", f"{huggingface_base}/pretrained/G32k.pth"),
        ("G40k.pth", f"{huggingface_base}/pretrained/G40k.pth"),
        ("G48k.pth", f"{huggingface_base}/pretrained/G48k.pth"),
        ("f0D32k.pth", f"{huggingface_base}/pretrained/f0D32k.pth"),
        ("f0D40k.pth", f"{huggingface_base}/pretrained/f0D40k.pth"),
        ("f0D48k.pth", f"{huggingface_base}/pretrained/f0D48k.pth"),
        ("f0G32k.pth", f"{huggingface_base}/pretrained/f0G32k.pth"),
        ("f0G40k.pth", f"{huggingface_base}/pretrained/f0G40k.pth"),
        ("f0G48k.pth", f"{huggingface_base}/pretrained/f0G48k.pth"),
    ]

    for filename, url in v1_models:
        run_aria2c(url, pretrained_dir, filename)

    # --- 2. Download Pretrained Models (v2) ---
    print("\n=== Mengunduh Pretrained Models V2 ===")
    pretrained_v2_dir = os.path.join(base_dir, "pretrained_v2")
    
    # Hanya model yang umum digunakan di v2 (sesuai snippet Anda yang di-uncomment)
    v2_models = [
        ("D32k.pth", f"{huggingface_base}/pretrained_v2/D32k.pth"),
        ("D40k.pth", f"{huggingface_base}/pretrained_v2/D40k.pth"),
        ("D48k.pth", f"{huggingface_base}/pretrained_v2/D48k.pth"), # Opsional
        ("G40k.pth", f"{huggingface_base}/pretrained_v2/G40k.pth"),
        ("G48k.pth", f"{huggingface_base}/pretrained_v2/G48k.pth"), # Opsional
        ("f0D40k.pth", f"{huggingface_base}/pretrained_v2/f0D40k.pth"),
        ("f0D32k.pth", f"{huggingface_base}/pretrained_v2/f0D32k.pth"), # Opsional
        ("f0G40k.pth", f"{huggingface_base}/pretrained_v2/f0G40k.pth"),
        ("f0G32k.pth", f"{huggingface_base}/pretrained_v2/f0G32k.pth"), # Opsional
        ("f0G48k.pth", f"{huggingface_base}/pretrained_v2/f0G48k.pth"), # Opsional
    ]

    for filename, url in v2_models:
        run_aria2c(url, pretrained_v2_dir, filename)

    # --- 3. Download UVR5 Weights ---
    print("\n=== Mengunduh Model UVR5 ===")
    uvr_dir = os.path.join(base_dir, "uvr5_weights")
    
    uvr_models = [
        ("HP2-人声vocals+非人声instrumentals.pth", f"{huggingface_base}/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth"),
        ("HP5-主旋律人声vocals+其他instrumentals.pth", f"{huggingface_base}/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth"),
        # Bisa tambahkan model lain seperti MDX-Net jika diperlukan
    ]

    for filename, url in uvr_models:
        run_aria2c(url, uvr_dir, filename)

    # --- 4. Download Hubert Base ---
    print("\n=== Mengunduh Hubert Base ===")
    # Hubert biasanya di root atau assets
    run_aria2c(f"{huggingface_base}/hubert_base.pt", base_dir, "hubert_base.pt")

    # --- 5. Download RMVPE Model ---
    print("\n=== Mengunduh RMVPE Model ===")
    run_aria2c(f"{huggingface_base}/rmvpe.pt", base_dir, "rmvpe.pt")

    print("\n" + "="*40)
    print("Semua aset telah selesai diunduh atau dicoba.")
    print("Silakan jalankan infer-web.py atau rvc_cli.py.")
    print("="*40)

if __name__ == "__main__":
    # Cek apakah aria2c terinstall
    try:
        subprocess.run(["aria2c", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        print("Error: aria2c tidak ditemukan. Silakan install dengan: pip install aria2 atau apt-get install aria2c")
        sys.exit(1)
    
    main()