for image in $(ls res/SetA); do
    echo $image
    python3 src/HLD_Helper.py 'res/SetA/'$image
    echo
done
