clc;
clear all;
close all;
load('SENNE_data_new.mat')
datasets_names={'SENNE_data_new'};
datasets={SENNE_data_new};
global dim;
global Global_Meta;
global levels;
global novel_buff;
global total_queury_ins;
global global_labels;
global PT_meta_all;
global gamma_all;
global Initial_TD_size;
global lambda;
global delT;
global known_acc;
global novel_acc;
global novel_buff_size;
global total_fp;
global curTime;
global Actual_Pred;
Initial_TD_size=2000;
novel_buff_size=250;
levels=5;
lambda=0.0005;%orginal 0.001 and T=0.1352(2000 decay)// 0.0005->0.1353 (decay 4000)
delT=0.1353;%0.1352;
%  f=0;% comment this for novel class discovery/ uncomment for
%  classification without novel classes
block_size=1000; % for real datasets set block size = 2000


for ds=1:length(datasets)
    
    data=datasets{ds};
    dsname=datasets_names{ds};
    
    %%%%%%%%%%
    %     r=randperm(size(data,1));
    %     data=data(r,:);
    %%%%%%%%%%%%%%%%
    label=data(:,end);
    data=data(:,1:end-1);
    Dlen=size(data,1);
    Initial_data=data(1:Initial_TD_size,:);
    Initial_labels=label(1:Initial_TD_size);
    
    [dataset_labels, ia1, cid1] = unique(label,'stable');
    dataset_counts = sum( bsxfun(@eq, cid1, unique(cid1)') )';
    
    [global_labels, ia, cid] = unique(Initial_labels);
    counts = sum( bsxfun(@eq, cid, unique(cid)') )';
    %%%%%%%%%%%%%%%file write%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Global_Meta={};
    total_queury_ins=0;
    PT_meta_all={};
    gamma_all=[];
    dim=size(data,2);
    
    
    known_acc=0;
    
    b=0;
    final_result=[];
    novel_acc=0;
    total_novel=0;
    novel_buff=[];
    
    total_fp=0;
    curTime=0;
    
    Actual_Pred=[];
    
    buffer=[];
    
    
    Initial_class_PT_Extraction([Initial_data Initial_labels]);
    fprintf('\n Initial_class_PT_Extraction Done');
    
    
    %streaming loop
    tic;
    for i=Initial_TD_size+1:Dlen
        c_ins=data(i,:);
        c_ins_label=label(i);
        curTime=i;
        
        [f]=detect_novel(c_ins,PT_meta_all);
%         f=0;% comment this for novel class discovery
        if ~ismember(c_ins_label,global_labels)
            total_novel=total_novel+1;
        end
        if f==-1
            novel_buff=[novel_buff;[c_ins,c_ins_label,b]];
        else
            
            [p_idx,pre_label,Dist]=classify(c_ins);
            Actual_Pred=[Actual_Pred;[c_ins_label,pre_label]];
            
            buffer=[buffer; [c_ins,pre_label,c_ins_label,0]];%0 is flg
            
            if c_ins_label==pre_label
                known_acc=known_acc+1;
                
            end
        end
        if size(novel_buff,1)>=novel_buff_size
            class_PT_Extraction(novel_buff(:,1:end-1),b);
            
            f_accc=(known_acc+novel_acc)*100/(i-Initial_TD_size);
            
            fprintf('\n Block accuracy =%f',f_accc);
        end
        
        if mod(i,block_size)==0
            %update model
            updateModel_buffer(buffer);
            buff_idx=find(b-novel_buff(:,end)==2);
            edata=novel_buff(buff_idx,1:end-1);
            classifydata(edata);
            novel_buff(buff_idx,:)=[];
            buffer=[];
            
            f_accc=(known_acc+novel_acc)*100/(i-Initial_TD_size);
            %
            
            fprintf('\n Block accuracy =%f',f_accc);
            b=b+1;
            fprintf('\n Block N0 =%d',b);
            
        end
        
        f_accc=(known_acc+novel_acc)*100/(i-Initial_TD_size);
        final_result=[final_result;f_accc];
        
        
    end% loop over all datasets
    
    t=toc;
    
    fprintf('\t Final time=%f  \n',t/60);
    if f==-1
        temp_label=novel_buff(:,end);
        Actual_Pred=[Actual_Pred;[temp_label,-1*ones(size(novel_buff,1),1)]];
        %%%%%%existing instances
        n_idx=~ismember(temp_label,global_labels);
        
        novel_acc=novel_acc+sum(n_idx);
        total_fp=total_fp+sum(~n_idx);
    end
    f_accc=(known_acc+novel_acc)*100/(i-Initial_TD_size);
    final_result=[final_result;f_accc];
    label_per=total_queury_ins*100/(i-Initial_TD_size);
    fprintf('\t Final accuracy=%f  \n',final_result(end));
    fprintf('\t Label Per =%f  \n',label_per);
    
end %data set loop


