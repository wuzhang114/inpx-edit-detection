function plot_tifs_matlab_v22
% Compact curves informed by DomainForensics, FoCus, MFCLIP and ReLoc.
root = fileparts(fileparts(mfilename('fullpath')));
out = fullfile(root,'paper','figures','matlab_v22');
if ~exist(out,'dir'), mkdir(out); end
d = jsondecode(fileread(fullfile(root,'outputs','refit_final.json')));
l = jsondecode(fileread(fullfile(root,'outputs','layer_phase_summary.json')));
e = jsondecode(fileread(fullfile(root,'outputs','layer_evidence.json')));
assert(max(abs(mean(d.loc_runs,1)' - d.loc_mean(:))) < 1e-10);
assert(max(abs(std(d.loc_runs,0,1)' - d.loc_sd(:))) < 1e-10);
blue = [0 114 178]/255; orange = [213 94 0]/255;
set(groot,'defaultFigureVisible','off','defaultTextInterpreter','none', ...
    'defaultAxesTickLabelInterpreter','none','defaultLegendInterpreter','none');
for language = ["en","zh"]
    zh = language == "zh";
    font = 'Arial'; if zh, font = 'Microsoft YaHei'; end
    f = canvas(2.12);
    a = panel(f,[.075 .25 .39 .59],font); b = panel(f,[.57 .25 .39 .59],font);
    for ax = [a b]
        xlim(ax,[-0.13 sx(14731)+0.17]); ylim(ax,[0 1]); yticks(ax,0:.2:1);
        xticks(ax,sx([0 10 200 1000 14731])); xticklabels(ax,{'0','10','200','1k','ALL'});
        xlabel(ax,pick(zh,'训练掩码预算 k','Training-mask budget, k'));
    end
    ylabel(a,'Micro IoU');
    ylabel(b,'AUROC');
    heading(a,pick(zh,'(a) 定位','(a) Localization'));
    heading(b,pick(zh,'(b) 检测','(b) Detection'));
    p = d.models.Log_sigmoid.params;
    k = [0 logspace(-2,log10(14731),500)];
    y = p(1)+p(2)./(1+exp(-p(3)*(log10(max(k,realmin))-p(4)))); y(1)=p(1);
    hfit = plot(a,sx(k),y,'--','Color',blue,'LineWidth',1.05);
    hm = errorbar(a,sx(d.budgets),d.loc_mean,d.loc_sd,'o','Color',blue, ...
        'LineWidth',.75,'MarkerSize',3.3,'MarkerFaceColor',blue,'CapSize',3);
    held = d.log_sigmoid_summary.heldout_reassessment;
    hh = plot(a,sx([held.k]),[held.measured],'d','LineStyle','none', ...
        'Color',orange,'MarkerFaceColor',orange,'MarkerSize',3.6);
    lg = legend(a,[hm hfit hh],{pick(zh,'均值 ± SD','Mean ± SD'), ...
        pick(zh,'拟合','Fit'),pick(zh,'原留出点','Held out')}, ...
        'Location','northwest','Box','off','FontSize',8,'FontName',font);
    lg.ItemTokenSize=[13 8];
    errorbar(b,sx(d.budgets),d.det_mean,d.det_sd,'s-','Color',orange, ...
        'LineWidth',1.05,'MarkerSize',3.3,'MarkerFaceColor',orange,'CapSize',3);
    saveplot(f,out,"budget_"+language);

    f=canvas(2.12);
    a=panel(f,[.075 .25 .235 .59],font);
    b=panel(f,[.405 .25 .235 .59],font);
    c=panel(f,[.735 .25 .235 .59],font);
    layers=[3 6 9 12]; vals=zeros(4,2); contrast=zeros(1,4); consistency=contrast;
    for i=1:4
        vals(i,1)=l.(sprintf('L%d_b200',layers(i))).miou;
        vals(i,2)=l.(sprintf('L%d_b1000',layers(i))).miou;
        q=e.(sprintf('x%d',layers(i))); contrast(i)=q.dist_edit; consistency(i)=q.dir_consist;
    end
    for ax=[a b c]
        xlim(ax,[2.4 12.6]); xticks(ax,layers);
        xlabel(ax,pick(zh,'读出层','Layer'));
    end
    plot(a,layers,vals(:,1),'-o','Color',blue,'LineWidth',1.05,'MarkerSize',3.3,'MarkerFaceColor',blue);
    plot(a,layers,vals(:,2),'-s','Color',orange,'LineWidth',1.05,'MarkerSize',3.3,'MarkerFaceColor',orange);
    ylim(a,[.28 .60]); yticks(a,.3:.1:.6);
    ylabel(a,'Micro IoU');
    heading(a,pick(zh,'(a) 定位','(a) Localization'));
    lg=legend(a,{'k = 200','k = 1,000'},'Location','northwest','Box','off','FontSize',8);
    lg.ItemTokenSize=[12 8];
    plot(b,layers,contrast,'-o','Color',[.28 .40 .48],'LineWidth',1.05,'MarkerSize',3.3,'MarkerFaceColor',[.28 .40 .48]);
    ylim(b,[.35 .85]); yticks(b,.4:.2:.8);
    ylabel(b,pick(zh,'特征对比度','Feature contrast'));
    heading(b,pick(zh,'(b) 区域对比度','(b) Regional contrast'));
    plot(c,layers,consistency,'-^','Color',[.28 .40 .48],'LineWidth',1.05,'MarkerSize',3.5,'MarkerFaceColor',[.28 .40 .48]);
    ylim(c,[.88 1.01]); yticks(c,[.90 .95 1]); ytickformat(c,'%.2f');
    ylabel(c,pick(zh,'余弦相似度','Cosine similarity'));
    heading(c,pick(zh,'(c) 方向一致性','(c) Direction consistency'));
    saveplot(f,out,"layers_"+language);
end
save(fullfile(out,'source_data.mat'),'d','l','e');
fprintf('MATLAB curves exported: %s\n',out);
end

function f=canvas(height)
f=figure('Color','w','Units','inches','Position',[1 1 7.16 height], ...
    'Renderer','painters','PaperPositionMode','auto');
end
function ax=panel(f,pos,font)
ax=axes(f,'Position',pos,'FontName',font,'FontSize',8, ...
    'LineWidth',.55,'Box','on','TickDir','in','TickLength',[.012 .012], ...
    'XColor',[.18 .18 .18],'YColor',[.18 .18 .18], ...
    'YGrid','off','XGrid','off','Layer','bottom');
hold(ax,'on');
end
function heading(ax,s)
text(ax,.5,1.075,s,'Units','normalized','FontName',ax.FontName,'FontSize',8.5, ...
    'FontWeight','normal','HorizontalAlignment','center','VerticalAlignment','bottom');
end
function y=sx(x)
% Same symlog coordinate as the original: linear <=10, log10 above 10.
y=x/10*(.55/(1-.1)); m=x>10; y(m)=.55/(1-.1)+log10(x(m)/10);
end
function s=pick(zh,c,en)
if zh,s=c;else,s=en;end
end
function saveplot(f,out,name)
drawnow;
exportgraphics(f,fullfile(out,name+'.pdf'),'ContentType','vector','BackgroundColor','white');
exportgraphics(f,fullfile(out,name+'.png'),'Resolution',300,'BackgroundColor','white');
savefig(f,fullfile(out,name+'.fig')); close(f);
end
