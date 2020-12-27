{
  description = "recut-pipeline";
  inputs = { 
    unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    # mcp3d.url = "git+ssh://git@muyezhu/mcp3d#kdm-dev";
    mcp3d.url = "/curr/kdmarrett/mcp3d";
  };

  outputs = inputs: {
    defaultPackage.x86_64-linux = import ./recut.nix { nixpkgs = inputs.unstable; mcp3d = inputs.mcp3d; };
  };
}
