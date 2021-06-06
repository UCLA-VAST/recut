<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" name="UNET_csim_design" top="top_kernel">
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="false" clean="false" ldflags="" mflags=""/>
    </Simulation>
    <files>
        <file name="../../UNet_tb.cpp" sc="0" tb="1" cflags=" -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="cnn_sw.cpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="cnn_sw.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="hw_kernel.cpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="util.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
    </files>
    <solutions>
        <solution name="solution1" status=""/>
    </solutions>
</AutoPilot:project>

