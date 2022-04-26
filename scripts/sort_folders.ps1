for ( $i = 0; $i -lt $args.count; $i++ ) {
    $dir = $args[$i]
    write-host "Sorting swcs in: $dir"
    $files = Get-ChildItem $dir 
    foreach ($f in $files) {
	$outfile = $f.FullName
	write-host "    Sorting $outfile ..."
	# all points will be connected automatically (default behavior) since /p parameter is not specified
	C:/Users/YangRecon2/Desktop/Vaa3D/Vaa3D_V3.601_Windows_MSVC_64bit/vaa3d_msvc.exe /x sort_neuron_swc /f sort_swc /i $outfile /o $outfile 
    }
}
