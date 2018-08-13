for image in $(ls res/SetA); do
    echo $image
    python3 src/ColorDetector.py 'res/SetA/'$image
    echo
done
