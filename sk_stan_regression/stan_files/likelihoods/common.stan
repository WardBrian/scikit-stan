/*   GENERAL LINK MAP 
     
     identity - 0
     log - 1
     inverse - 2
     sqrt - 3
     1/mu^2 - 4
     logit - 5
     probit - 6
     cloglog - 7
     cauchit - 8

                     */

// TODO: add discrete. maybe separate discrete/continuous? 
vector common_invert_link(vector mu, int link) { 
    if (link == 0) { // identity
        return mu;
    } 
    else if (link == 1) { // log link  
        return exp(mu); 
    } 
    else if (link == 2) { // inverse link 
        return inv(mu); 
    }
    else if (link == 3) { // sqrt link  
        return square(mu); 
    }
    else {  // 1/mu^2 link
        return inv_sqrt(mu);
    } 

    
}
