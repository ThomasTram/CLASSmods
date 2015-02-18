clear; clc;
close all;
%Plot Z files

%figure('PaperType','a4',...
%'PaperPositionMode','manual',...
%'PaperOrientation','landscape',...
%'PaperUnits','centimeters',...
%'PaperPosition',[1,1,28,19]);

H = 9;
W = 7;
figure('Position',[200,100,400,600],...
'PaperUnits','inches',...
'PaperOrientation','portrait',...
'PaperSize',[W,H],...
'PaperPosition',[0,0,W,H])
maxl = 5

fname = 'Helvetica';

for dolog = [true,false]

for mcase = ['0','m']

for l = 0:maxl
  

  filename = ['Z',mcase,'_',num2str(l,'%03d'),'.dat'];

  Zl = load(filename);

  dq = 20/(length(Zl)-1);
  q = 1e-1+dq*(0:99);

  subplot(3,2,l+1)
  
  %contourf(q,q,Zl)
  if dolog
    imagesc(q,q,log10(abs(Zl)));set(gca,'Ydir','normal');
  else
    imagesc(q,q,Zl);set(gca,'Ydir','normal');  
  end
  colorbar('Fontname',fname)
  set(gca,'FontName',fname);
  %imagesc(q,q,Zl);set(gca,'Ydir','normal');

  if dolog
    title(['log_{10}(|Z^',mcase,'_',num2str(l),'|)'],'FontName',fname)
  else
    title(['Z^',mcase,'_',num2str(l)],'FontName',fname)
  end
  
  xlabel('q\prime','FontName',fname)
  ylabel('q','FontName',fname)
  %colorbar
end

if dolog
  pdfname = ['Zl',mcase,'_log.pdf']
else
  pdfname = ['Zl',mcase,'.pdf']
end

saveas(gcf,pdfname)
end

end