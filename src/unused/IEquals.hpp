#ifndef EQUALS_HPP
#define EQUALS_HPP

#include <functional>

/** ===========================================================================================================================
 * Class similar to a functional interface used to compare two generic values and decide if they are equal or not
 * 
 * It is basically a binary predicate
 */
template <class T> class Equals{

    private:

    const std::function<bool(T, T)> comparatorDouble; // The comparing function

    public:

    /** -----------------------------------------------------------------------------------------------------------------------
     * Equals() - Public constructor that takes in the comparing function that is used in the operator() overload
     * 
     * @param compareFunction - a function that takes two parameters and return bool deciding about equality of its arguments
     */
    Equals(const std::function<bool(T, T)>& compareFunction);
    
    /** -----------------------------------------------------------------------------------------------------------------------
     * operator() - Public overload that takes two generic values and calls the compareFunction to return the bool
     * 
     * @param a - first operand to compare
     * @param b - second operand to compare
     * 
     * @return - bool true if a and b are the same and false if they are not the same
     */
    bool operator()(T a, T b) const;
    bool operator()(T* a, T* b) const;

};

#endif