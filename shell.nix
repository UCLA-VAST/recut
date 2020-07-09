# https://discourse.nixos.org/t/how-do-i-build-a-nix-shell-which-depends-on-some-unstable-packages/928/2
let
  unstable = import (fetchTarball https://nixos.org/channels/nixos-unstable/nixexprs.tar.xz) { };
in
{ pkgs ? import <nixpkgs> {} }:
with pkgs; mkShell {
  buildInputs = [ openssl boost libtiff
    clang-tools python38Packages.pandas python38Packages.plotly python38Packages.matplotlib unstable.cmake
    unstable.mpich];
}

