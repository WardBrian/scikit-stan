/* Common functions used for various components of the package. */

/* Inverse of the Cauchit link function; the inverse of the CDF for the Cauchy
    distribution. 

@param mu: linear predictor to be inverted under the cauchit
@return vector of linear predictors after the Cauchit link inversion  
*/
vector inv_cauchit(vector mu) { 
    return atan(mu) / pi() + 0.5; 
}

/*   GENERAL LINK MAP 
     
     identity - 0
     log - 1
     inverse - 2
     sqrt - 3
     inverse-square - 4
     logit - 5
     probit - 6
     cloglog - 7
     cauchit - 8

                     */
                     
/* Helper function that performs link function inversion. See the table above
    for the mapping between link function and internal numerical representation. 

Validation of a family-link function pair is assumed to occur outside of the function. 

This is a match of the R Families package: 
https://stat.ethz.ch/R-manual/R-devel/library/stats/html/family.html

@param mu: linear predictor vector to be inverted
@param link: internal numerical representation of link function
@return vector of the linear predictor after inverse link has been applied ("unlinked")
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
    else if (link == 4) {  // inverse-square link
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
