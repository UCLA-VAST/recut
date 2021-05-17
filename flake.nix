{
  description = "recut";
  inputs = {
    pinned.url = "github:NixOS/nixpkgs/733e537a8ad76fd355b6f501127f7d0eb8861775";
    # mcp3d fails on latest unstable hdf5 API changes
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    mcp3d.url = "git+ssh://git@github.com/muyezhu/mcp3d?ref=stable_lib";
    #openvdb.url = "git+ssh://git@github.com/UCLA-VAST/openvdb?ref=pointalias";
    openvdb.url = "git+ssh://git@github.com/UCLA-VAST/openvdb";
    # alternatively you could pin a certain commit like:
    # mcp3d.url = "git+ssh://git@github.com/muyezhu/mcp3d?ref=kdm-dev&rev=<commit hash>";
    # or you could use your local filesystem branch with:
    # mcp3d.url = "/home/kdmarrett/mcp3d";

    # pin nix package manager versions to exact match between mcp3d and recut
    mcp3d.inputs.nixpkgs.follows = "pinned";
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
