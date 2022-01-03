import os
valiable =  ["0000686","0009948","0011127","0009652","0008489","0010485","0007024"]
outdir = 'compare_225'
os.system(f'mkdir -p {outdir}')
for i in valiable:
  os.system(f"cp output/NgocHuyen_delightfulTTS/result/225000/{i}.wav {outdir}/{i}-Delightful.wav")