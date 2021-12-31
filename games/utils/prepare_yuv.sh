# @echo off
# set back=%cd%
ROOT_SOURCE="/home/ola/Documents/PhD/reinforcement/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/games/utils/meta/split/"

TARGET_DIR="/home/ola/Documents/PhD/reinforcement/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/games/utils/meta/yuv/"

cd $ROOT_SOURCE
for d in */ ; do
    d=$(echo $d | sed 's:/*$::')
    echo "$d"
    ffmpeg -i $d/%06d.jpg -s 1024x640 -pix_fmt yuv420p -r 200 "$TARGET_DIR$d.yuv"
done




# ffmpeg -i dataset/%03d.jpg -s 1024x640 -pix_fmt yuv420p -r 200 f.yuv
# ffplay -f rawvideo -pixel_format yuv420p -s 1024x640 -i f.yuv 