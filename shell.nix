{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [ pkgs.openssl pkgs.boost pkgs.libtiff
    pkgs.clang-tools pkgs.python38Packages.pynvim pkgs.nodejs pkgs.cmake ];
}

