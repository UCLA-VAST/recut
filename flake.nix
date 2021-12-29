{
  description = "recut";
  inputs = {
    # pick latest commit from stable branch and test it so no suprises
    nixpkgs.url = "github:NixOS/nixpkgs/d53978239b265066804a45b7607b010b9cb4c50c";
    mcp3d.url = "git+ssh://git@github.com/ucla-brain/mcp3d";
    openvdb.url = "git+ssh://git@github.com/UCLA-VAST/openvdb";

    # pin nix package manager versions to exact match between mcp3d and recut
    mcp3d.inputs.nixpkgs.follows = "nixpkgs";
    openvdb.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = inputs: {

    defaultPackage.x86_64-linux = import ./recut.nix {
      nixpkgs = inputs.nixpkgs;
      mcp3d = inputs.mcp3d;
      openvdb = inputs.openvdb;
    };
  };
}
