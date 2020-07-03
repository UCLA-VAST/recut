{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [ pkgs.openssl pkgs.boost pkgs.libtiff
    pkgs.clang-tools pkgs.python38Packages.pandas pkgs.python38Packages.plotly pkgs.python38Packages.matplotlib pkgs.cmake ];
}

