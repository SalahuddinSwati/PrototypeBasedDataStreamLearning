function [memory_win]=updateWin(buffer,memory_win,label_per)
        KK=size(buffer,1);
    num_replicates=1;
    [membership, ctrs] = kmeans(memory_win(:,1:end-2),KK,'Replicates',num_replicates,'Distance','sqEuclidean');
    temp=[]; 
    for j=1:KK
        clu_pt=find(membership==j);
        clu_data=memory_win(clu_pt,:);
        clu_label=clu_data(find(clu_data(:,end)==1),end-1);
        if length(clu_label)>1
           [unicl_lb, ia, cid] = unique(clu_label,'stable');
             counts = sum( bsxfun(@eq, cid, unique(cid)') )';
             [~,id]=max(counts);
             clu_label=unicl_lb(id);
        end
        if ~isempty(clu_label)
            temp=[temp;[ctrs(j,:),clu_label,1]];
        else
            temp=[temp;[ctrs(j,:),0,0]];
        end
    end
     [buffer]=Extract_PT(buffer);
     memory_win=[temp;buffer];
end