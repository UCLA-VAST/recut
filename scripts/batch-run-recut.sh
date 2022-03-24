
echo "Finding dirs in: $1"

EXT_PATH=pipeline_v.3.0/ch0/morph_neurite+soma/neurite+soma_segmentation/segmentation/
for d in $1/*/ ; do
    DIR=`basename $d`
    mkdir -p $DIR
    echo $DIR
    cd $DIR
    cp -r $d/$EXT_PATH/marker_files .

    # create point.vdb
    if [ ! -f point.vdb ]
    then
    recut $d/$EXT_PATH --convert --input-type tiff --output-type point --parallel 8
    else
	echo "   found vdb"
    fi

    # if [ ! -f components ]
    # then
    #recut point.vdb --seeds marker_files --output-windows uint8.vdb --run-app2
    # else
    #	echo "   found previous"
    # fi
    cd ..
done
