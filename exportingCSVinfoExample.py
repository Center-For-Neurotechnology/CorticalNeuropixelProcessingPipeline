

import bpy

outputFile = 'D:/Dropbox/ACPProjects/PtNRElectrodes/Neuropixel/Pt03Range2Part1.csv'

####verts = [ bpy.context.object.matrix_world * v.co for v in bpy.context.object.data.vertices ]

verts = [ v.co for v in bpy.context.object.data.vertices ]

csvLines = [ ";".join([ str(v) for v in co ]) + "\n" for co in verts ]

f = open( outputFile, 'w' )
f.writelines( csvLines )
f.close()