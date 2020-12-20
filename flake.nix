{
  description = "recut-pipeline";
  inputs = { 
    unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    #mcp3d.url = "ignore";
    mcp3d.url = "/home/kdmarrett/mcp3d";
  };

  outputs = inputs: {
    defaultPackage.x86_64-linux = import ./recut.nix { nixpkgs = inputs.unstable; mcp3d_path = inputs.mcp3d.url; };
  };
}
