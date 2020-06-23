{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [ pkgs.openssl pkgs.boost pkgs.libtiff ];
}
