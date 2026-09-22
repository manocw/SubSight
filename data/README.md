# SubPipe dataset — download instructions (Chunk0 only for now)
#
# 1. Go to https://zenodo.org/doi/10.5281/zenodo.10053564 and download the
#    Chunk0 archive. Docs: https://github.com/remaro-network/SubPipe-dataset
# 2. Unzip so you end up with:
#
#      data/
#        Chunk0/
#          Segmentation/
#            <timestamp>.png          <- input image
#            <timestamp>_label.png    <- mask (Stage 2 will decode classes)
#
# 3. Do NOT commit raw data to git (see ../.gitignore). It is large and
#    licensed CC-BY-4.0 — check the Zenodo page for reuse/citation rules.
#
# 4. Colab tip: upload Chunk0 to Google Drive once, then mount it:
#
#      from google.colab import drive
#      drive.mount('/content/drive')
#      # then unzip / copy Chunk0/Segmentation into ./data/
#
# Placeholder so the folder exists in git:
