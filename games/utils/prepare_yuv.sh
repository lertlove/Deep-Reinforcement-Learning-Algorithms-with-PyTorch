# @echo off
# set back=%cd%
ROOT_SOURCE="/mnt/nas/openImageNet/metadata/list_files/split/"

TARGET_DIR="/mnt/nas/openImageNet/metadata/list_files/yuv/"

cd $ROOT_SOURCE
for d in */ ; do
    d=$(echo $d | sed 's:/*$::')
    echo "$d"
    ffmpeg -i $d/%06d.jpg -s 1024x640 -pix_fmt yuv420p -r 200 "$TARGET_DIR$d.yuv"
done




# ffmpeg -i dataset/%03d.jpg -s 1024x640 -pix_fmt yuv420p -r 200 f.yuv
# ffplay -f rawvideo -pixel_format yuv420p -s 1024x640 -i f.yuv 