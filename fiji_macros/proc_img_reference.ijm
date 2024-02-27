open();
run("Invert LUT");
makeRectangle(0, 10, 775, 365);
run("Enhance Contrast", "saturated=0.35");
run("Set Scale...", "distance=1 known=0.016 unit=µm");
run("Scale Bar...", "width=1 height=1 thickness=4 font=14 color=White background=Black location=[Lower Right] horizontal bold overlay");
saveAs("PNG");
close();