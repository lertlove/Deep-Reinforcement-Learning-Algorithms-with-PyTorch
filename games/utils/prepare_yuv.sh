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

### Resolution
# 416x240
# 832x480

# 1920x1080 -> 480x270
# 1280x720 -> 320x180

# 1024x768 -> 256x192

# 2560x1600 -> 640x400

###Frame Rate
# 20,24,30,50,60

###QP
# 22,27,32,37