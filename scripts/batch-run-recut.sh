
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
    recut $d/$EXT_PATH --convert point.vdb --parallel 24
    else
	echo "   found vdb"
    fi

    if [ ! -f uint8.vdb ]
    then
    recut $d/$EXT_PATH --convert uint8.vdb --type uint8 --parallel 24
    else
	echo "   found vdb"
    fi

    if [ ! -f float.vdb ]
    then
    recut $d/$EXT_PATH --convert float.vdb --type float --parallel 24
    else
	echo "   found vdb"
    fi

    # if [ ! -f components ]
    # then
    recut point.vdb --seeds marker_files --output-windows uint8.vdb --run-app2
    # else
    #	echo "   found previous"
    # fi
    cd ..
done
