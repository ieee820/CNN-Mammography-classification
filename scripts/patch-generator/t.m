amgh = @(gl) (log10(gl+eps) - 4.80662)/-1.07553;
ahow = @(gl)3.789 - 0.00094568 * gl;
bclum = @(gl) ( gl - 4096.99 ) / -1009.01;
dhow = @(gl) 3.96604095240593 + (-0.00099055807612) * (gl);

sat = 3.5;

v_oda = @(od) 1 - od / sat;

ad1 = '../DDSM/cases/normals/normal_02/case0200/A_0200_1.LEFT_CC.png'; 
ad2 = '../DDSM/cases/normals/normal_08/case4600/D_4600_1.RIGHT_MLO.png';
ad3 = '../DDSM/cases/cancers/cancer_01/case0001/C_0001_1.RIGHT_CC.png';
ad4 = '../DDSM/cases/normals/normal_11/case1955/A_1955_1.RIGHT_CC.png';

im = cell(3,1);
od = cell(3,1);
v_od = cell(3,1);

im{1} = double(imread(ad1));
od{1} = amgh(im{1});
od{1}(od{1} > sat) = sat;
v_od{1} = v_oda(od{1});

im{2} = double(imread(ad2));
od{2} = dhow(im{2});
od{2}(od{2} > sat) = sat;
v_od{2} = v_oda(od{2});

im{3} = double(imread(ad3));
od{3} = bclum(im{3});
od{3}(od{3} > sat) = sat;
v_od{3} = v_oda(od{3});

im{4} = double(imread(ad4));
od{4} = ahow(im{4});
od{4}(od{4} > sat) = sat;
v_od{4} = v_oda(od{4});

min(im{4}(:))
max(im{4}(:))

if 0
    imagesc(im{3}); colormap(gray)
else
    for i = 1:4
        subplot(2,2,i)
        imagesc(v_od{i}); colormap(gray)
    end
end