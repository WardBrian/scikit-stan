/* Inverse Cauchit link 

*/
// TODO: add this to Stan core math?  
vector inv_cauchit(vector mu) { 
    return atan(mu) / pi() + 0.5; 
}

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
/* Helper function that performs link function inversion. See the table above
    for the mapping between link function and internal numerical representation. 

@param mu:
@param link: internal numerical representation of link function
*/
vector common_invert_link(vector mu, int link) {
    // NOTE: this assumes validated family-link combinations 
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
    else if (link == 4) {  // 1/mu^2 link
        return inv_sqrt(mu);
    } 
    else if (link == 5) { // logit link
        return inv_logit(mu); 
    }
    else if (link == 6) { // probit link  
        return Phi(mu);
    }
    else if (link == 7) { // cloglog link
        return inv_cloglog(mu);
    }
    else { // cauchit link  
        return inv_cauchit(mu);
    }   
}
