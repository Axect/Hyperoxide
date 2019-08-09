pub trait FP {
    type Element;

    fn fmap<F>(&self, f: F) -> Self
    where
        F: Fn(Self::Element) -> Self::Element;
    fn zip_with<G>(&self, g: G, other: &Self) -> Self
    where
        G: Fn(Self::Element, Self::Element) -> Self::Element;
    fn reduce<G>(&self, g: G, default: Self::Element) -> Self::Element
    where
        G: Fn(Self::Element, Self::Element) -> Self::Element;
}
