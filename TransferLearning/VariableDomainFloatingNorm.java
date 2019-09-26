package TransferLearning;

import burlap.mdp.core.state.vardomain.VariableDomain;

/** A tuple specifying the numeric domain of a {@link burlap.mdp.core.state.State} variable.
 *  Based on the code from @author James MacGlashan.
 *  This version includes the possibility of specifying the interval on which one wants to normalize
 *  his values by setting @var newLower and @var newUpper. This can help performing better knowledge 
 *  transfer: normally one would normalize in [0,1] but, when using tile coding we would have, for 
 *  instance, one tile (or multiple ones) spanning the values [0,1), and another one (or a new set of
 *  them) for the only value 1. This implementation is suppose to help in this undesiderable situation*/
public class VariableDomainFloatingNorm extends VariableDomain{

	/** The upper and lower value of the new domain interval */
	public double newLower = 0.;
	public double newUpper = 0.99;
	
	/** Initializes.
	 *  @param lower The lower value of the domain
	 *  @param upper The upper value of the domain */
	public VariableDomainFloatingNorm(double lower, double upper) {
		this.lower = lower;
		this.upper = upper;
	}

	/** Given a value in this variable domain, returns its normalized value. That is,
	 *  (d - lower) / (upper - lower)
	 *  @param d the input value
	 *  @return the normalized value */
	@Override
	public double norm(double d){
		double norm = newLower + ((d - lower)*(newUpper - newLower) / this.span());
		if(norm < newLower || norm > newUpper) {
			norm = 1.;
		}
					
		return norm;
	}
}
