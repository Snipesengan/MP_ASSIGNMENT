for image in $(ls res/SetA); do
    echo $image
    python3 src/HLD_Pipeline.py 'res/SetA/'$image
    echo
done
