{
  description = "recut";
  inputs = {
    # pick latest commit from stable branch and test it so no suprises
    nixpkgs.url = "github:NixOS/nixpkgs/d53978239b265066804a45b7607b010b9cb4c50c";
    openvdb.url = "github:UCLA-VAST/openvdb?ref=feat/reachable-resurfacing-nix";
    gel.url = "github:UCLA-VAST/GEL?ref=experimental2";

    # pin nix package manager versions to exact match to recut
    openvdb.inputs.nixpkgs.follows = "nixpkgs";
    gel.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = inputs: {

    defaultPackage.x86_64-linux = import ./recut.nix {
      nixpkgs = inputs.nixpkgs;
      openvdb = inputs.openvdb;
      gel = inputs.gel;
    };

    # optionally output a docker container
    docker = let
      recut = import ./recut.nix {
        nixpkgs = inputs.nixpkgs;
        openvdb = inputs.openvdb;
        gel = inputs.gel;
      };
      # pkgs = inputs.nixpkgs;
      pkgs = import <nixpkgs> {};
    in pkgs.dockerTools.buildLayeredImage {
      name = "kdmarrett/recut";
      tag = recut.version;
      contents = [ recut ];

      config = {
        Cmd = [ "recut" ];
        WorkingDir = "/";
      };
    };

  };
}
