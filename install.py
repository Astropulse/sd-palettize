import launch
import os

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        urlparts = lib.split("#egg=")
        if len(urlparts) > 1:
            packname = urlparts[1]
        else:
            packname = "hitherdither"
        if not launch.is_installed(packname):
            launch.run_pip(f"install {lib}", f"sd-palettize requirement: {lib}")
