### Resolution
RESOLUTION="416x240"
RATE=50
INTERVAL=$((1000/$RATE)) #rate=50 -> interval = 1000/rate

ROOT_SOURCE="/mnt/nas/openImageNet/metadata/list_files/split/"

TARGET_DIR="/mnt/nas/openImageNet/metadata/list_files/yuv/${RESOLUTION}_${RATE}/"

mkdir $TARGET_DIR

NUM_DIR=`ls $ROOT_SOURCE |wc -l`
cd $ROOT_SOURCE

for ((d=11;d<$NUM_DIR;d++)); 
do 
   # your-unix-command-here
   echo $d
   ffmpeg -i $d/%06d.jpg -s $RESOLUTION -pix_fmt yuv420p -r $INTERVAL "$TARGET_DIR$d.yuv"
   wait
done


# printf '%s\0' */ | sort -zV | while read -rd '' d; do 
#     d=$(echo $d | sed 's:/*$::')
#     echo "$d"
    # ffmpeg -i $d/%06d.jpg -s $RESOLUTION -pix_fmt yuv420p -r $INTERVAL "$TARGET_DIR$d.yuv"
    # wait
# done

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