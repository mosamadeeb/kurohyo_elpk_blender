# kurohyo_elpk_blender
Blender Import addon for Kurohyo PSP games ELPK container files. Can import models, textures, and animations. Supports Blender versions >= 2.93 and <= 3.2.
 
***
 
# Installing
Download the [latest release](https://github.com/SutandoTsukai181/kurohyo_elpk_blender/releases/latest) and install it in Blender. To do so, follow the instructions in the [official Blender manual](https://docs.blender.org/manual/en/latest/editors/preferences/addons.html) for installing add-ons, or follow the brief instructions below.

Open the `Edit` -> `Preferences` window from the menu bar, go to `Add-ons`, click on the `Install` button, and select the release zip you downloaded. Then, enable the script by checking the box next to it.

# Setting up the addon
In order to import skeletons, the path to `globals.bin` from Kurohyo 1 or `skeleton.bin` from Kurohyo 2 must be set in the addon preferences.

After installing the addon, you should be able to see the preferences under the addon's name in Blender preferences. You should select a game type and set the path to the file (`either globals.bin` for KH1 or `skeleton.bin` for KH2).

![image](https://user-images.githubusercontent.com/52977072/182150260-febc57b6-292a-4577-97d8-5e7716fc34c6.png)

***

# Usage
In Blender, go to File -> Import -> **Kurohyo ELPK Container**

Textures will be loaded automatically when loading a model, if the textures file is in the same directory as the model.
In order to import animations, a compatible armature from Kurohyo must be selected first.

***

# Credits
Thanks to Capitan Retraso for [rELPKckr](https://github.com/CapitanRetraso/rELPKckr).

Thanks to Violet for helping with the formats and for testing.

***

## Please credit this addon properly when using content created using the addon. This applies to mods, model/animation ports, and even MMD videos.
