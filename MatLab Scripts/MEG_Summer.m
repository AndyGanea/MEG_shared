%% PARAMETERS
Method    = {'gPDC','iPDC'};           
Frequency = [10,20,100];               
Subject   = {'DOC','GB','JDC','JFXD','JZ','LT','NvA','RR','SJB','BG'};
Conds     = {'pro_L','anti_L','pro_R','anti_R'};

% Node labels
baseNodes = {'V1','V3','SPOC','AG','POJ','SPL','mIPS','VIP','IPL','STS','S1','M1','SMA','PMd','FEF','PMv'};
nodeNames = [ strcat(baseNodes,'L'), strcat(baseNodes,'R') ];

% Paths
data_dir      = "C:\Users\dillo\OneDrive\Documents\MEG Summer\MEG_20250620_cue_vols_626_to_1563_order_10\Sta_adj_wd_order_10";
save_dir_main = "C:\Users\dillo\OneDrive\Documents\MEG Summer\Saved_Heatmaps";
if ~exist(save_dir_main,'dir'), mkdir(save_dir_main); end

% Build diverging colormap: blue→white→red
n      = 256;
half   = n/2;
blue2w = [linspace(0,1,half)', linspace(0,1,half)', ones(half,1)];
w2red  = [ones(half,1), linspace(1,0,half)', linspace(1,0,half)'];
divCmap = [blue2w; w2red];

%% MAIN LOOPS
for mi = 1:numel(Method)
  method = Method{mi};
  method_dir = fullfile(save_dir_main,method);
  if ~exist(method_dir,'dir'), mkdir(method_dir); end

  for fi = 1:numel(Frequency)
    freq = Frequency(fi);
    fprintf(">> %s, %dHz\n",method,freq);

    %% PART A: Load & compute per-subject sensory code
    subj_codes = cell(numel(Subject),1);
    for si = 1:numel(Subject)
      subj = Subject{si};

      % Accumulate condition sums
      avgC = struct(); cntC = struct();
      for c = Conds
        avgC.(c{1}) = []; cntC.(c{1}) = 0;
      end

      % Scan folders under the data directory
      dList = dir(data_dir);
      folderNames = {dList([dList.isdir]).name};
      folderNames = folderNames(~ismember(folderNames,{'.','..'}));
      
      % Filter: keep only folders with this condition, and exclude LT
      for c = Conds
        cond = c{1};
        for k = 1:length(folderNames)
          folder = folderNames{k};
          if contains(folder,'LT'), continue; end  % skip LT trials
          if contains(folder,cond)
            fld = fullfile(data_dir, folder);
            fn  = sprintf("%s_%s_%dHz_alg1_crit1_cat.csv",subj,method,freq);
            fp  = fullfile(fld,fn);
            if exist(fp,'file')
              A = readmatrix(fp); A(isnan(A))=0;
              if isempty(avgC.(cond))
                avgC.(cond) = zeros(size(A));
              end
              avgC.(cond) = avgC.(cond) + A;
              cntC.(cond) = cntC.(cond) + 1;
            end
          end
        end
      end

      % Finalize averages
      for c = Conds
        if cntC.(c{1})>0
          avgC.(c{1}) = avgC.(c{1})/cntC.(c{1});
        else
          avgC.(c{1}) = [];
        end
      end

      % SensoryCode = (pro_L+anti_L)-(pro_R+anti_R)
      if all(~cellfun(@(x) isempty(avgC.(x)),Conds))
        S = (avgC.pro_L + avgC.anti_L) - (avgC.pro_R + avgC.anti_R);
      else
        S = [];
        warning("Subject %s missing conditions", subj);
      end
      subj_codes{si} = S;
    end

    %% PART B: Global scale for individual maps
    allvals = [];
    for k = 1:numel(subj_codes)
      S = subj_codes{k};
      if ~isempty(S)
        allvals = [allvals; S(:)];
      end
    end
    if isempty(allvals)
      warning("No data for %s %dHz", method, freq);
      continue;
    end
    cmax_global = max(abs(allvals),[],'omitnan');

    %% PART C: Per-subject heatmaps
    for si = 1:numel(Subject)
      subj = Subject{si};
      S    = subj_codes{si};
      if isempty(S), continue; end

      S(1:end+1:end) = NaN;
      outd = fullfile(method_dir, sprintf("%dHz",freq), "Subject_"+subj);
      if ~exist(outd,'dir'), mkdir(outd); end

      h = heatmap(S,'XDisplayLabels',nodeNames,'YDisplayLabels',nodeNames);
      h.MissingDataColor = [1 1 1];
      colormap(divCmap);
      caxis([-cmax_global cmax_global]);
      title(sprintf("%s | %s | Sensory=(pL+aL)-(pR+aR) | %dHz", subj, method, freq));
      saveas(gcf, fullfile(outd,subj+"_SensoryCode.png"));
      close;
    end

    %% PART D: Group average and t-test correction
    valid = ~cellfun(@isempty,subj_codes);
    GA    = mean(cat(3,subj_codes{valid}),3);
    GA(1:end+1:end) = NaN;

    grpdir = fullfile(method_dir, sprintf("%dHz",freq), "Group_Average");
    if ~exist(grpdir,'dir'), mkdir(grpdir); end

    % raw group scale
    rawvals = GA(:);
    rawvals = rawvals(~isnan(rawvals));
    cmax_group = max(abs(rawvals),[],'omitnan');

    % Raw group heatmap
    h = heatmap(GA,'XDisplayLabels',nodeNames,'YDisplayLabels',nodeNames);
    h.MissingDataColor = [1 1 1];
    colormap(divCmap);
    caxis([-cmax_group cmax_group]);
    title(sprintf("Group | %s | Sensory=(pL+aL)-(pR+aR) | %dHz", method, freq));
    saveas(gcf, fullfile(grpdir,"Group_SensoryCode.png"));
    close;

    % T-test correction
    [n,n2] = size(GA);
    Corr = nan(n,n2);
    for i=1:n
      for j=1:n2
        if i~=j
          vals = cellfun(@(x)x(i,j), subj_codes(valid));
          [~,p] = ttest(vals,0,'Alpha',0.05);
          if p<0.05
            Corr(i,j)=GA(i,j);
          end
        end
      end
    end
    Corr(1:end+1:end)=NaN;

    % corrected group scale
    corrvals = Corr(:);
    corrvals = corrvals(~isnan(corrvals));
    if isempty(corrvals)
      cmax_corr = cmax_group/2;
    else
      cmax_corr = max(abs(corrvals),[],'omitnan');
    end

    % Corrected heatmap
    h = heatmap(Corr,'XDisplayLabels',nodeNames,'YDisplayLabels',nodeNames);
    h.MissingDataColor = [1 1 1];
    colormap(divCmap);
    caxis([-cmax_corr cmax_corr]);
    title(sprintf("Group Corrected | %s | Sensory=(pL+aL)-(pR+aR) | %dHz", method, freq));
    saveas(gcf, fullfile(grpdir,"Group_SensoryCode_Corrected.png"));
    close;

  end
end

disp("Processing complete for all methods and frequencies.");
