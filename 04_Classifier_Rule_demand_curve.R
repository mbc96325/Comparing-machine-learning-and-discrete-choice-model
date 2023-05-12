
rm(list=ls(all=TRUE)) 
library(caret)
library(faraway)
library(car)

# read file
setwd("C:/Users/Baichuan/Dropbox (MIT)/00_Research/04_Others_share/project_ml_tb_benchmark/code/")



Model_run <- function(data, data_test, id_col_cost, num_points, MODE_WANT,  Dependent_var, method_list, Base_acc, Re_run, output_file_path){
  
  Results = data.frame()
  
  for (method in method_list){
    
    tryCatch(
      # This is what I want to do...
      {
        # if (Re_run != 1){
        #   if (any(Results[,'Model']==method)){
        #     print(paste0(method, " exists, skip it"))
        #     next
        #   }
        # }
        
        
        fml = as.formula(paste0(Dependent_var, " ~ ."))
        print(paste0('===Training ', method, ' ...==='))
        
        model = train(fml, data = data, method = method)
        
        
        
        min_X = 0#min(data_test[, id_col_cost])
        max_X = 1#max(data_test[, id_col_cost])
        
        all_points = seq(from=min_X, to = max_X, by = (max_X -min_X)/(num_points-1))
        
        prop_list = c()
        
        #rv1 <- 
        #rv2 <- append(rv1, 50, after = 40)
        
        
        for (point_value in all_points){
          x_temp = data_test
          x_temp[,id_col_cost] = point_value
          
          model_pred = predict(model, newdata = x_temp, type="prob")
          y_prob = model_pred[, MODE_WANT]
          prop = mean(y_prob)
          
          # model_pred = predict(model, newdata = x_temp, type="raw")
          # y_pred = as.numeric(model_pred)
          # prop = sum(y_pred == MODE_WANT)/length(y_pred)
          
          prop_list = append(prop_list, prop)
        }
        model_list = rep(method, num_points)
        
        #Results = Results.append(pd.DataFrame({'Model': [name] * self.num_points, 'x_values':all_points,'pred_prop':prop_list}))
        #prop_list = rep(0.46, num_points)
        
        df = data.frame(Model = model_list,x_values = all_points,pred_prop = prop_list)
        Results = rbind(Results, df)
        
        
        # start_time = Sys.time()
        # fml = as.formula(paste0(Dependent_var, " ~ ."))
        # print(paste0('===Training ', method, ' ...==='))
        # model = train(fml, data = data, method = method,
        #               trControl = trainControl(method = "cv",number=5))
        # end_time = Sys.time()
        # time_taken = as.numeric(end_time - start_time, units = "secs")
        # #=========Save data========
        # temp_record = model$resample
        # temp_record$Model = method
        # temp_record$base = Base_acc
        # temp_record$Run_time_5CV_second = round(time_taken,2)
        # temp_record$Computer_info = computer_info
        # temp_record$n_jobs = n_jobs
        # colnames(temp_record)[which(names(temp_record) == "Resample")] <- "Fold"
        # temp_record = temp_record[, -which(names(temp_record) %in% c("Kappa"))]
        # #==========
        # Results = rbind(Results, temp_record)
        # # add avg
        # avg_acc = mean(temp_record$Accuracy)
        # temp_avg<-data.frame(method, "Average", avg_acc, Base_acc, computer_info, n_jobs, round(time_taken,2))
        # names(temp_avg)<-c("Model",'Fold','Accuracy','base',
        #                    'Computer_info','n_jobs','Run_time_5CV_second')
        # Results = rbind(Results, temp_avg)
        # #==========
        # 
        # # save every iteration
        # Results <- Results[c("Model",'Fold','Accuracy','base',
        #                      'Computer_info','n_jobs','Run_time_5CV_second')] 
        write.csv(Results, file = output_file_path, row.names=FALSE)
        
      },
      error=function(error_message) {
        print(paste0(method, " is not applicable"))
      }
    )
    
  }
  
}


###########Define methods here (CRret)###################################

method_list = c("C5.0Rules")

#########################################################################

seed = 1
set.seed(seed)
sample_size_list = c('10k')
#sample_size_list = c('100k')
computer_info = 'I9-9900K'
Re_run = 1 # 1: rerun all models, 0: if there are results existed, jump it
n_jobs = 1

Estimator_name = 'Rule_R'

Dependent_var_list = c('MODE')
#Dependent_var_list = c('TRIPPURP')

DATASET_list = c('London')




col_cost = 'cost_driving_ccharge' # 'cost_driving_fuel' # dur_pt_bus cost_driving_ccharge
num_points = 100

MODE_WANT = 4 #4 # drive  1: walk, 2: cycle, 3: public transport, 4:drive
# (1: walk, 2: cycle, 3: public transport, 4:drive)



for (DATASET in DATASET_list){
  for (Dependent_var in Dependent_var_list){
    if (DATASET == 'London' || DATASET == 'SG'){
      if (Dependent_var != 'MODE'){
        next
      }
      
    }
    if (Dependent_var == 'MODE'){
      output_name = 'MC'
      data_name = 'mode_choice'
    } else if (Dependent_var == 'CAR_OWN'){
      output_name = 'CO'
      data_name = 'car_ownership'
    } else{
      output_name = 'TP'
      data_name = 'trip_purpose'
    }
    for (sample_size in sample_size_list){
      if (DATASET == 'SG'){
        if (sample_size == '100k'){
          next
        }
      }
      
      if (DATASET == 'London'){
        output_file_path = paste0('Demand_curve_results/Results_London_', output_name, '_',
                                  Estimator_name, '_', sample_size, '.csv')
        data_name_read = paste0('train_data_London.csv')
        data_test_read = paste0('test_data_London.csv')
        data = read.csv(paste0('London_dataset/', data_name_read),header=TRUE, sep = ",", stringsAsFactors = FALSE)
        data_test = read.csv(paste0('London_dataset/', data_test_read),header=TRUE, sep = ",", stringsAsFactors = FALSE)
        
        
      } else if (DATASET == 'SG'){
        output_file_path = paste0('Results/Results_SG_', output_name, '_',
                                  Estimator_name, '_', sample_size, '.csv')
        data_name_read = paste0('data_SG_', data_name, '_', sample_size, '.csv')
        data = read.csv(paste0('SG_dataset/', data_name_read),header=TRUE, sep = ",", stringsAsFactors = FALSE)
      }
      else{
        output_file_path = paste0('Results/Results_', output_name, '_',
                                  Estimator_name, '_', sample_size, '.csv')
        data_name_read = paste0('data_', data_name, '_', sample_size, '.csv')
        data = read.csv(paste0('data/', data_name_read),header=TRUE, sep = ",", stringsAsFactors = FALSE)
      }
      
      Base_acc = as.numeric(sort(table(data[,Dependent_var]),decreasing=TRUE)[1]/nrow(data))
      
      if (Re_run == 1){
        Results = setNames(data.frame(matrix(ncol = 7, nrow = 0)), c("Model",'Fold','Accuracy','base',
                                                                     'Computer_info','n_jobs','Run_time_5CV_second'))
      }else{
        if (!file.exists(output_file_path)){
          Results = setNames(data.frame(matrix(ncol = 7, nrow = 0)), c("Model",'Fold','Accuracy','base',
                                                                       'Computer_info','n_jobs','Run_time_5CV_second'))		
        }else{
          Results = read.csv(output_file_path,header=TRUE, sep = ",", stringsAsFactors = FALSE)
          
        }
        
      }
      
      col_names_list = colnames(data)
      #col_names_list = col_names_list[col_names_list != "MODE"]
      
      id_col_cost = which(col_names_list==col_cost)
      
      #data[,id_col_cost]
      
      data[,Dependent_var]<-factor(data[,Dependent_var])
      data_test[,Dependent_var]<-factor(data_test[,Dependent_var])
      
      new_tail = paste0('_', col_cost, '_' , 'mode', toString(
        MODE_WANT), '.csv')
      output_file_path = gsub(".csv",new_tail, output_file_path)
      
      
      Model_run(data, data_test,id_col_cost, num_points, MODE_WANT, Dependent_var, method_list, Base_acc, Re_run, output_file_path)
      
    }
    
  }
}


  


