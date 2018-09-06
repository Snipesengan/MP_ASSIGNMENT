file="res/SetA"

for image in $(ls $file); do
    echo $image
    python3 src/HLD_Pipeline.py $file'/'$image
    echo
done
