{ pkgs ? import (builtins.fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-20.03.tar.gz") {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.openssl
    pkgs.boost
    pkgs.python38Packages.matplotlib 
    pkgs.cmake
    #pkgs.libtiff
    #pkgs.mpich
  ];
}

