let
  sources = import ./nix/sources.nix;
  pkgs = import sources.nixpkgs {};
in
pkgs.stdenv.mkDerivation {
  name = "recut";

  # https://nixos.org/nix/manual/#builtin-filterSource
  src = builtins.filterSource 
    (path: type: pkgs.lib.cleanSourceFilter path type 
    && baseNameOf path != "build"
    && baseNameOf path != "bin/*"
    && baseNameOf path != "data/*") ./.;

  cmakeFlags = ["-DFROM_NIX_BUILD=ON"];
  nativeBuildInputs = [ pkgs.cmake ];

  # used for automated github testing 
  # see .github/workflows/*.yaml
  doCheck = true;
  enableParallelBuilding = true;

  buildInputs = [ 
    pkgs.python38Packages.matplotlib 
    pkgs.gtest
    pkgs.gbenchmark
    # warning leaving breakpointHook on 
    # can create many sleeping processes on your system
    # it will cause github actions to hang, if there are any failures
    #pkgs.breakpointHook
    #openssl 
    #boost 
    #libtiff 
    #unstable.mpich 
    #clang_10 
    #llvmPackages.openmp 
  ];

  buildPhase = "make -j 8 && make install -j 8";

  # an environment variable which can be used within CMakelists or source
  #IN_NIX_BUILD_ENV="ON";

  # You have to run the install step first such that the relevant files used at runtime
  # are in the data directory before running any other tests
  checkPhase="pwd; ls ../; echo; ls ../data; ./recut_test --gtest_filter=Install.\"*\"; ./recut_test --gtest_output=json:../data/test_detail.json | tee ../data/test_detail.log";

  installPhase = ''
    mkdir -p $out/bin
    cp /build/recut-pipeline/bin/recut_test $out/bin/recut_test
  '';
}
