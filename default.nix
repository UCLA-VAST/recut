{
  pkgs   ? import <nixpkgs> {},
         stdenv ? pkgs.stdenv
}:
rec {
  myProject = stdenv.mkDerivation {
    name = "recut";
    version = "0.1";

    src = ./.;
    checkPhase = ''
      make -C test check
      make -C benchmark check
      '';

    installPhase = ''
      mkdir -p $out/include
      cp -r include/attoparsecpp $out/include/
      '';

    buildInputs = with pkgs; [
# (callPackage ./googlebench.nix { stdenv = stdenv; })
    ];
  };
}
