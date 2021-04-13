final: prev: {
  openvdb = prev.stdenv.mkDerivation {
    version = "8.0.1";
    name="openvdb";
    src = prev.fetchFromGitHub {
      owner = "AcademySoftwareFoundation";
      repo = "openvdb";
      rev = "v8.0.1";
      sha256 = "0qzx6l5c183k6j9zki31gg9aixf5s1j46wdi7wr1h3bz7k53syg9";
    };

    cmakeFlags =["-DOPENVDB_BUILD_VDB_VIEW=ON -DOPENVDB_BUILD_VDB_RENDER=ON -DOPENVDB_BUILD_VDB_LOD=ON"];

    nativeBuildInputs = with prev; [ cmake pkg-config ];
    buildInputs = with prev; [
      unzip openexr boost tbb jemalloc c-blosc ilmbase 
      # for the optional VDB_VIEW binary opengl related dependencies:
      libGL glfw3 x11 libGLU xorg.libXdmcp
    ];
  };
}
