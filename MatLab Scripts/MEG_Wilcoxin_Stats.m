%% PARAMETERS
Methods     = {'gPDC','iPDC'};
Frequencies = [10,20,100];
Subjects    = {'DOC','GB','JDC','JFXD','JZ','LT','NvA','RR','SJB','BG'};
Conds       = {'pro_L','anti_L','pro_R','anti_R'};

% Node labels
baseNodes = {'V1','V3','SPOC','AG','POJ','SPL','mIPS','VIP','IPL','STS','S1','M1','SMA','PMd','FEF','PMv'};
nodeNames = [ strcat(baseNodes,'L'), strcat(baseNodes,'R') ];

% Paths
data_dir      = "C:\Users\dillo\OneDrive\Documents\MEG Summer\MEG_20250620_cue_vols_626_to_1563_order_10\Sta_adj_wd_order_10";
save_base_dir = "C:\Users\dillo\OneDrive\Documents\MEG Summer\Saved_Heatmaps";
if ~exist(save_base_dir,'dir'), mkdir(save_base_dir); end

ignorePattern = 'LT';  % exclude any folder containing "LT"

%% LOOP OVER METHODS & FREQUENCIES
for mi = 1:numel(Methods)
  method = Methods{mi};
  method_dir = fullfile(save_base_dir, method);
  if ~exist(method_dir,'dir'), mkdir(method_dir); end

  allResults = table( ...
    'Size',[0 6], ...
    'VariableTypes',{'string','double','string','string','double','double'}, ...
    'VariableNames',{'Method','Frequency','Region_from','Region_to','W_stat','p_value'} ...
  );

  for fi = 1:numel(Frequencies)
    freq = Frequencies(fi);

    % --- PART 1: load & compute per-subject sensory codes ---
    codes = cell(numel(Subjects),1);
    for si = 1:numel(Subjects)
      subj = Subjects{si};
      % condition accumulators
      avgC = struct(); cntC = struct();
      for c = Conds
        avgC.(c{1}) = []; cntC.(c{1}) = 0;
      end

      % find all candidate folders
      D = dir(data_dir);
      folders = {D([D.isdir]).name};
      folders = folders(~ismember(folders,{'.','..'}));

      % load each condition
      for c = Conds
        for k = 1:numel(folders)
          fldname = folders{k};
          if contains(fldname,ignorePattern), continue; end
          if contains(fldname,c{1})
            fp = fullfile(data_dir,fldname,sprintf("%s_%s_%dHz_alg1_crit1_cat.csv",subj,method,freq));
            if exist(fp,'file')
              A = readmatrix(fp); A(isnan(A))=0;
              if isempty(avgC.(c{1}))
                avgC.(c{1}) = zeros(size(A));
              end
              avgC.(c{1}) = avgC.(c{1}) + A;
              cntC.(c{1}) = cntC.(c{1}) + 1;
            end
          end
        end
      end

      % finalize averages
      for c = Conds
        if cntC.(c{1})>0
          avgC.(c{1}) = avgC.(c{1}) / cntC.(c{1});
        else
          avgC.(c{1}) = [];
        end
      end

      % sensory code
      if all(~cellfun(@(x) isempty(avgC.(x)),Conds))
        codes{si} = (avgC.pro_L + avgC.anti_L) - (avgC.pro_R + avgC.anti_R);
      else
        codes{si} = [];
        warning("Missing data subj %s",subj);
      end
    end

    % pick a template matrix for size
    template = codes{ find(~cellfun(@isempty,codes),1) };
    if isempty(template)
      warning("No valid data for %s %dHz",method,freq);
      continue;
    end
    [nNodes,~] = size(template);

    % --- PART 2: signrank for each pair i≠j ---
    for i = 1:nNodes
      for j = 1:nNodes
        if i==j, continue; end

        % gather subject values at (i,j)
        vals = cellfun(@(M) M(i,j), codes, 'UniformOutput', true);
        vals = vals(~isnan(vals));  % drop NaNs

        if isempty(vals)
          W = NaN; p = NaN;
        elseif all(vals==0)
          W = 0;     % no difference
          p = 1;     
        else
          [p,~,stats] = signrank(vals, 0, 'alpha', 0.05);
          W = stats.signedrank;
        end

        % append to table
        allResults = [ allResults; 
          { method, freq, nodeNames{i}, nodeNames{j}, W, p } ];
      end
    end
  end

  % --- PART 3: save to CSV ---
  outFile = fullfile(method_dir, sprintf("signrank_results_%s.csv",method));
  writetable(allResults, outFile);
  fprintf("Saved %s\n", outFile);
end

disp("Done.");
