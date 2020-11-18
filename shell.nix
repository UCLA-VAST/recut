let
  sources = import ./nix/sources.nix;
  pkgs = import sources.nixpkgs {};
in
pkgs.mkShell {
  buildInputs = [
    pkgs.openssl
    pkgs.boost
    pkgs.python38Packages.matplotlib 
    pkgs.cmake
    pkgs.gdb
    pkgs.clang-tools
    #pkgs.libtiff
    #pkgs.mpich
  ];
}

