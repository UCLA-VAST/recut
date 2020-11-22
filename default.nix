let
  sources = import ./nix/sources.nix;
  pkgs = import sources.nixpkgs {};
in
  with pkgs;
stdenv.mkDerivation {
  name = "recut";

  # https://nixos.org/nix/manual/#builtin-filterSource
  src = builtins.filterSource 
    (path: type: lib.cleanSourceFilter path type 
    && baseNameOf path != "build"
    && baseNameOf path != "bin/*"
    && baseNameOf path != "data/*") ./.;

  cmakeFlags = ["-DFROM_NIX_BUILD=ON"];
  nativeBuildInputs = [ cmake ];

  # used for automated github testing 
  # see .github/workflows/*.yaml
  doCheck = true;
  enableParallelBuilding = true;

  buildInputs = [ 
    python38Packages.matplotlib 
    gtest
    gbenchmark
    range-v3
    # warning leaving breakpointHook on 
    # can create many sleeping processes on your system
    # it will cause github actions to hang, if there are any failures
    # always comment it out before pushing
    breakpointHook
  ];

  # You have to run the install step first such that the relevant files used at runtime
  # are in the data directory before running any other tests
  checkPhase="./recut_test --gtest_filter=Install.\"*\"; ./recut_test --gtest_output=json:../data/test_detail.json | tee ../data/test_detail.log";

  installPhase = ''
    mkdir -p $out/bin
    cp /build/recut-pipeline/build/recut_test $out/bin/recut_test
  '';
}
